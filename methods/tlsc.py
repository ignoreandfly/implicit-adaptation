"""Thought-level self-certainty maximisation (arXiv 2603.07159).

Repeat until the answer ends: sample K candidate next "thoughts", score each by its average
token self-certainty KL(U || p) = -1/V sum_j log(V p_j) (Kang et al. 2025; averaged over the
candidate's own tokens, conditioned on the prompt + chosen prefix), keep the best (Eq. 3).

Adaptations (documented, the paper leaves them open or targets chat models):
  - thought boundary: "." or newline (the paper splits on blank lines; base-model plain-format
    answers are one paragraph and end at a blank line)
  - candidate temperature --temperature (not given in the paper; default 0.7), top-p 1.0
  - search only in the first --tlsc_steps thoughts, greedy afterwards (paper's ablation found
    the first 1-5 steps matter most); min/max thought length 1/300 tokens, max 40 thoughts,
    total budget --max_new_tokens tokens of chosen text
"""

import math

import torch

from common import prompts
from methods.common import encode, make_path, prefix_slug

THOUGHT_STOPS = [".", "\n"]


def tag(args):
    return f"K{args.tlsc_k}_steps{args.tlsc_steps}_T{args.temperature}" + \
        (f"_prefix-{prefix_slug(args.prefix)}" if args.prefix else "")


def run(model, tok, texts, answers, task, args, ends):
    K, V = args.tlsc_k, model.config.vocab_size
    res = []
    for b in range(0, len(texts), args.batch_problems):
        base = [t + args.prefix for t in texts[b:b + args.batch_problems]]
        B = len(base)
        text, used, spent, done = [""] * B, [0] * B, [0] * B, [False] * B
        for step in range(40):
            act = [i for i in range(B) if not done[i]]
            if not act:
                break
            k = K if step < args.tlsc_steps else 1
            budget = max(1, min(300, args.max_new_tokens - min(used[i] for i in act)))
            enc = encode(tok, [base[i] + text[i] for i in act], args.fmt)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=budget, do_sample=k > 1,
                                     temperature=args.temperature if k > 1 else None, top_k=0 if k > 1 else None,
                                     top_p=1.0 if k > 1 else None, num_return_sequences=k,
                                     pad_token_id=tok.pad_token_id, return_dict_in_generate=True,
                                     output_logits=k > 1, stop_strings=THOUGHT_STOPS, tokenizer=tok)
            seqs = out.sequences[:, enc.input_ids.shape[1]:].tolist()
            if k > 1:
                sc = torch.stack([(-torch.log_softmax(l.float(), -1).mean(-1) - math.log(V)) for l in out.logits], 1)
            for a, i in enumerate(act):
                best, best_sc, best_ids, best_end = None, -1e9, None, False
                for j in range(k):
                    ids = seqs[a * k + j]
                    n = next((t for t, x in enumerate(ids) if x in ends), len(ids))
                    hit_eos = n < len(ids) and ids[n] != tok.pad_token_id or (n < len(ids) and ids[n] == tok.eos_token_id)
                    ids = ids[:n]
                    spent[i] += max(1, n)
                    s = float(sc[a * k + j, :max(1, n)].mean()) if k > 1 else 0.0
                    if s > best_sc:
                        best, best_sc, best_ids, best_end = j, s, ids, hit_eos
                piece = tok.decode(best_ids, skip_special_tokens=True)
                text[i] += piece
                used[i] += len(best_ids)
                if (best_end or not best_ids or used[i] >= args.max_new_tokens
                        or (args.fmt == "plain" and any(s in text[i] for s in prompts.STOP_STRINGS))):
                    done[i] = True
        for i in range(B):
            p = make_path(tok, [], task, answers[b + i], args.max_new_tokens, prefix=args.prefix.lstrip() + text[i], fmt=args.fmt)
            p["gen_tokens"], p["truncated"] = used[i], used[i] >= args.max_new_tokens
            res.append({"paths": [p], "chosen": 0, "generated_tokens": spent[i]})
        print(f"{b + B}/{len(texts)}", flush=True)
    return res


def summarize(results, answers):
    n = len(results)
    return {"acc_chosen": sum(r["paths"][0]["score"]["lenient_correct"] for r in results) / n,
            "mean_generated_tokens": sum(r["generated_tokens"] for r in results) / n}
