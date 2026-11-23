"""DoLa: Decoding by Contrasting Layers (Chuang et al. 2024), own implementation.

At each step: pick the premature layer (from a bucket) whose early-exit distribution has the
largest JSD from the final layer; next-token score = log p_final - log p_premature, restricted
to the plausible set {x : p_final(x) >= alpha * max p_final}; then greedy.

Buckets follow the HF implementation for L <= 40: low = 0,2,..,L/2-2 ; high = L/2,..,L-2
(hidden_states[0] = embeddings). Early exit applies the final norm before lm_head.
Repetition penalty (paper recommends 1.2) is applied to generated tokens only.
"""

import math

import torch

from methods.common import apply_repetition_penalty, make_path, manual_decode, single_path_summary


def tag(args):
    return f"{args.dola_layers}_rep{args.rep_penalty}"


def _jsd(lp, lq):
    p, q = lp.exp(), lq.exp()
    m = (0.5 * (p + q)).clamp_min(1e-30).log()
    return 0.5 * (p * (lp - m)).sum(-1) + 0.5 * (q * (lq - m)).sum(-1)


def run(model, tok, texts, answers, task, args, ends):
    L = model.config.num_hidden_layers
    layers = list(range(0, L // 2, 2)) if args.dola_layers == "low" else list(range(L // 2, L, 2))
    norm, head = model.model.norm, model.lm_head
    log_alpha = math.log(args.dola_alpha)
    picked = []

    def step(logits, hs, gen):
        lf = torch.log_softmax(logits, -1)
        cand = torch.stack([torch.log_softmax(head(norm(hs[l])).float(), -1) for l in layers])  # C,B,V
        j = torch.stack([_jsd(lf, c) for c in cand])  # C,B
        best = j.argmax(0)
        picked.extend([layers[i] for i in best.tolist()])
        lp = cand[best, torch.arange(lf.shape[0], device=lf.device)]
        s = lf - lp
        s[lf < lf.max(-1, keepdim=True).values + log_alpha] = -float("inf")
        return apply_repetition_penalty(s, gen, args.rep_penalty).argmax(-1)

    res = []
    bs = args.batch_problems * 4
    for b in range(0, len(texts), bs):
        batch = texts[b:b + bs]
        gen = manual_decode(model, tok, batch, args.max_new_tokens, step, ends, need_hidden=True, fmt=args.fmt)
        for i, ids in enumerate(gen):
            res.append({"paths": [make_path(tok, ids, task, answers[b + i], args.max_new_tokens, fmt=args.fmt)], "chosen": 0})
        print(f"{b + len(batch)}/{len(texts)}", flush=True)
    from collections import Counter
    print("premature layer usage:", Counter(picked).most_common(), flush=True)
    return res


summarize = single_path_summary
