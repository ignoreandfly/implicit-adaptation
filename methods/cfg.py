"""Classifier-free guidance for LMs (Sanchez et al., ICML 2024, "Stay on topic with CFG").

    log p_hat(w) = log p(w | uncond) + gamma * (log p(w | cond) - log p(w | uncond))

cond   = full prompt  ("Question: q\\nAnswer:" + prefix)
uncond = prompt with the question removed ("Answer:" + prefix); both streams share the
         generated tokens. gamma = 1 is plain greedy; the paper uses gamma around 1.25-2.
Greedy decoding on p_hat. Both streams run in one batch (rows 0..B-1 cond, B..2B-1 uncond).
"""

import torch

from methods.common import make_path, manual_decode, prefix_slug, single_path_summary


def tag(args):
    return f"g{args.cfg_gamma}" + (f"_prefix-{prefix_slug(args.prefix)}" if args.prefix else "")


def run(model, tok, texts, answers, task, args, ends):
    g = args.cfg_gamma
    res = []
    bs = args.batch_problems * 4
    for b in range(0, len(texts), bs):
        cond = [t + args.prefix for t in texts[b:b + bs]]
        unc = [t[t.rfind("Answer:"):] + args.prefix for t in texts[b:b + bs]]
        B = len(cond)

        def step(logits, hs, gen):
            lp = torch.log_softmax(logits, -1)
            s = lp[B:] + g * (lp[:B] - lp[B:])
            nxt = s.argmax(-1)
            return torch.cat([nxt, nxt])

        gen = manual_decode(model, tok, cond + unc, args.max_new_tokens, step, ends)
        for i in range(B):
            res.append({"paths": [make_path(tok, gen[i], task, answers[b + i], args.max_new_tokens,
                                            prefix=args.prefix.lstrip())], "chosen": 0})
        print(f"{b + B}/{len(texts)}", flush=True)
    return res


summarize = single_path_summary
