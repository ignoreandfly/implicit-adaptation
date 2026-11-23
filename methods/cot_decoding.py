"""CoT-decoding (Wang & Zhou 2024, "Chain-of-Thought Reasoning Without Prompting").

For each question: take the top-k candidates for the FIRST generated token, continue each
greedily, and score every path by its answer confidence

    Delta = mean over answer tokens of  p(top-1) - p(top-2)

Selection rules reported:
    greedy   path 0 (the ordinary greedy decode; sanity check against the baseline)
    max      path with the highest Delta                      (paper's main rule)
    agg      answer with the highest summed Delta over paths  (paper's aggregated rule)
    oracle   any path correct (upper bound for any selection rule over these k paths)
"""

import re
from collections import defaultdict

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from common import prompts
from methods.common import encode, gen_stop_strings

_NUM_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")


class MarginRecorder(LogitsProcessor):
    """Records p(top1) - p(top2) at every decoding step; leaves the scores unchanged."""

    def __init__(self):
        self.steps = []

    def __call__(self, input_ids, scores):
        top = scores.float().softmax(-1).topk(2, dim=-1).values
        self.steps.append((top[:, 0] - top[:, 1]).cpu())
        return scores


def _answer_confidence(tok, ids, margins, text_full, cut, answer, normalize):
    """Mean margin over the tokens that spell the final answer in text_full[:cut].
    ids[0] is the branched first token (no margin); margins[j] belongs to ids[j + 1]."""
    if answer is None:
        return 0.0
    span = None
    for m in _NUM_RE.finditer(text_full[:cut]):
        if normalize(m.group()) == answer:
            span = (m.start(), m.end())
    if span is None:
        return 0.0
    vals, prev = [], 0
    for i in range(len(ids)):
        end = len(tok.decode(ids[: i + 1], skip_special_tokens=True))
        if i > 0 and end > span[0] and prev < span[1]:
            vals.append(margins[i - 1])
        prev = end
        if prev >= span[1]:
            break
    return float(sum(vals) / len(vals)) if vals else 0.0


def _cut_index(text, fmt="plain"):
    cut = len(text)
    if fmt != "plain":
        return cut
    for s in prompts.STOP_STRINGS:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    return cut


def run(model, tok, texts, answers, task, args, ends):
    k = args.k
    results = []
    for b in range(0, len(texts), args.batch_problems):
        batch = texts[b:b + args.batch_problems]
        enc = encode(tok, batch, args.fmt)
        with torch.no_grad():
            first = model(**enc).logits[:, -1].float().topk(k, dim=-1).indices  # (B, k)
        B = len(batch)
        ids = enc.input_ids.repeat_interleave(k, 0)
        att = enc.attention_mask.repeat_interleave(k, 0)
        ids = torch.cat([ids, first.reshape(-1, 1)], 1)
        att = torch.cat([att, torch.ones_like(att[:, :1])], 1)
        rec = MarginRecorder()
        with torch.no_grad():
            gen = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, pad_token_id=tok.pad_token_id,
                                 logits_processor=LogitsProcessorList([rec]),
                                 stop_strings=gen_stop_strings(args.fmt), tokenizer=tok)
        new = gen[:, ids.shape[1]:].tolist()
        margins = torch.stack(rec.steps, 1).tolist() if rec.steps else [[] for _ in new]
        for p in range(B):
            paths = []
            for j in range(k):
                r = p * k + j
                n = next((t for t, x in enumerate(new[r]) if x in ends), len(new[r]))
                path_ids = [first[p, j].item()] + new[r][:n]
                full = tok.decode(path_ids, skip_special_tokens=True)
                out = prompts.truncate(full, args.fmt)
                score = task.verify(out, answers[b + p])
                conf = _answer_confidence(tok, path_ids, margins[r][:n], full, _cut_index(full, args.fmt),
                                          score["lenient_answer"], task.normalize_number)
                paths.append({"output": out, "first_token": tok.decode(path_ids[:1]), "delta": conf,
                              "gen_tokens": n + 1, "truncated": n >= args.max_new_tokens, "score": score})
            best = max(range(k), key=lambda j: paths[j]["delta"])
            agg = defaultdict(float)
            for x in paths:
                if x["score"]["lenient_answer"] is not None:
                    agg[x["score"]["lenient_answer"]] += x["delta"]
            agg_ans = max(agg, key=agg.get) if agg else None
            results.append({"paths": paths, "chosen": best, "agg_answer": agg_ans})
        print(f"{min(b + len(batch), len(texts))}/{len(texts)}", flush=True)
    return results


def summarize(results, answers):
    n = len(results)
    def acc(f):
        return sum(f(r, a) for r, a in zip(results, answers)) / n
    return {
        "acc_greedy_path": acc(lambda r, a: r["paths"][0]["score"]["lenient_correct"]),
        "acc_max_delta": acc(lambda r, a: r["paths"][r["chosen"]]["score"]["lenient_correct"]),
        "acc_agg_delta": acc(lambda r, a: r["agg_answer"] == a),
        "acc_oracle": acc(lambda r, a: any(x["score"]["lenient_correct"] for x in r["paths"])),
        "mean_tokens_all_paths": sum(x["gen_tokens"] for r in results for x in r["paths"]) / n,
    }
