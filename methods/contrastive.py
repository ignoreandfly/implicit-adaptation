"""Contrastive decoding for reasoning (O'Brien & Lewis 2023) with a single-model amateur
(dropout amateur from Distillation Contrastive Decoding, Phan et al. 2024).

    score(w) = (1 + beta) * log p_expert(w) - beta * log p_amateur(w),
    restricted to {w : p_expert(w) >= alpha * max p_expert}; greedy on score.

Expert = the model as is. Amateur = the same model and prompt with dropout (rate --dropout)
applied to every decoder layer's output. (DCD also feeds the amateur invalid CoT
demonstrations; we are zero-shot, so only the dropout part is used.)
Defaults follow O'Brien & Lewis: beta = 0.5, alpha = 0.1.
"""

import math

import torch
import torch.nn.functional as F

from methods.common import make_path, manual_decode, prefix_slug, single_path_summary


def tag(args):
    return f"b{args.cd_beta}_drop{args.dropout}" + (f"_prefix-{prefix_slug(args.prefix)}" if args.prefix else "")


def run(model, tok, texts, answers, task, args, ends):
    beta, log_a = args.cd_beta, math.log(args.cd_alpha)
    state = {"B": 0}

    def hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        B = state["B"]
        h = torch.cat([h[:B], F.dropout(h[B:], args.dropout, training=True)])
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    handles = [layer.register_forward_hook(hook) for layer in model.model.layers]
    res = []
    bs = args.batch_problems * 4
    try:
        for b in range(0, len(texts), bs):
            batch = [t + args.prefix for t in texts[b:b + bs]]
            B = state["B"] = len(batch)

            def step(logits, hs, gen):
                lp = torch.log_softmax(logits, -1)
                e, a = lp[:B], lp[B:]
                s = (1 + beta) * e - beta * a
                s[e < e.max(-1, keepdim=True).values + log_a] = -float("inf")
                nxt = s.argmax(-1)
                return torch.cat([nxt, nxt])

            gen = manual_decode(model, tok, batch + batch, args.max_new_tokens, step, ends)
            for i in range(B):
                res.append({"paths": [make_path(tok, gen[i], task, answers[b + i], args.max_new_tokens,
                                                prefix=args.prefix.lstrip())], "chosen": 0})
            print(f"{b + B}/{len(texts)}", flush=True)
    finally:
        for h in handles:
            h.remove()
    return res


summarize = single_path_summary
