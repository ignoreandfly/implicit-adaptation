"""n samples per problem (HF generate). Selection rules are evaluated by methods.rescore.

uniform:     temperature --temperature everywhere
fork-aware:  --fork_threshold H (nats): temperature --t_high where the next-token entropy > H,
             --t_low elsewhere (Wang et al. 2025: high-entropy "fork" tokens steer reasoning).
"""

from collections import Counter

import torch
from transformers import LogitsProcessor, LogitsProcessorList

from common import prompts
from methods.common import prefix_slug, make_path


class ForkTemperature(LogitsProcessor):
    def __init__(self, thr, t_low, t_high):
        self.thr, self.t_low, self.t_high = thr, t_low, t_high
        self.hits, self.steps = 0, 0

    def __call__(self, input_ids, scores):
        lp = scores.float().log_softmax(-1)
        H = -(lp.exp() * lp).sum(-1)
        fork = H > self.thr
        self.hits += int(fork.sum())
        self.steps += fork.numel()
        T = torch.where(fork, torch.full_like(H, self.t_high), torch.full_like(H, self.t_low))
        return scores / T[:, None].to(scores.dtype)


def tag(args):
    p = ("prefix-" + prefix_slug(args.prefix) + "_") if args.prefix else ""
    if args.fork_threshold is not None:
        return f"{p}fork_thr{args.fork_threshold}_lo{args.t_low}_hi{args.t_high}_n{args.n}"
    return f"{p}T{args.temperature}_n{args.n}"


def run(model, tok, texts, answers, task, args, ends):
    proc = None
    if args.fork_threshold is not None:
        proc = ForkTemperature(args.fork_threshold, args.t_low, args.t_high)
    texts = [t + args.prefix for t in texts]
    res = []
    for b in range(0, len(texts), args.batch_problems):
        batch = texts[b:b + args.batch_problems]
        enc = tok(batch, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, do_sample=True, max_new_tokens=args.max_new_tokens,
                                 temperature=1.0 if proc else args.temperature, top_k=0, top_p=1.0,
                                 num_return_sequences=args.n, pad_token_id=tok.pad_token_id,
                                 logits_processor=LogitsProcessorList([proc]) if proc else None,
                                 stop_strings=prompts.STOP_STRINGS, tokenizer=tok)
        new = gen[:, enc.input_ids.shape[1]:].tolist()
        for p in range(len(batch)):
            paths = []
            for j in range(args.n):
                ids = new[p * args.n + j]
                n = next((t for t, x in enumerate(ids) if x in ends), len(ids))
                paths.append(make_path(tok, ids[:n], task, answers[b + p], args.max_new_tokens, prefix=args.prefix.lstrip()))
            votes = Counter(x["score"]["lenient_answer"] for x in paths if x["score"]["lenient_answer"] is not None)
            top = votes.most_common(1)[0][0] if votes else None
            chosen = next((j for j, x in enumerate(paths) if x["score"]["lenient_answer"] == top), 0)
            res.append({"paths": paths, "chosen": chosen})
        print(f"{b + len(batch)}/{len(texts)}", flush=True)
    if proc:
        print(f"fork steps: {proc.hits}/{proc.steps} = {proc.hits / max(1, proc.steps):.3f}", flush=True)
    return res


def summarize(results, answers):
    n = len(results)
    k = len(results[0]["paths"])
    return {"acc_majority": sum(r["paths"][r["chosen"]]["score"]["lenient_correct"] for r in results) / n,
            "acc_single_sample": sum(x["score"]["lenient_correct"] for r in results for x in r["paths"]) / (n * k),
            "acc_oracle": sum(any(x["score"]["lenient_correct"] for x in r["paths"]) for r in results) / n,
            "mean_tokens_all_paths": sum(x["gen_tokens"] for r in results for x in r["paths"]) / n}
