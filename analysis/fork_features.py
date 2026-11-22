"""Which GENERAL property of a choice at a fork token separates good from bad reasoning?

Data: N samples per GSM8K training problem from one model (e.g. SFT), labelled good/bad
(--label majority: agrees with the sample majority; gold: correct). Each sample is teacher-forced
through the same model with all hidden states. At every fork position (next-token entropy in the
top 20%), features of the token the sample actually chose:

  rank          rank of the chosen token under the model (1 = greedy)
  logp_gap      log p(top-1) - log p(chosen)              (0 for greedy choices)
  settle_depth  first layer l such that the chosen token is the logit-lens argmax at l and every
                later layer ("deep-thinking" at the token level; 1..L, L = only the last layer)
  late_agree    fraction of the last 4 layers whose logit-lens argmax is the chosen token
  next_entropy  entropy of the next-token distribution AFTER the choice (local certainty)
  next_margin   p(top-1) - p(top-2) after the choice

Per feature: AUROC for "chosen at a fork in a good sample" vs "in a bad sample" (0.5 = no signal),
and the same restricted to non-greedy choices (rank >= 2), which is what a fork rule has to judge.
A feature with clear signal defines a general single-pass rule: at a fork, pick among the top-k
candidates the one maximizing (or minimizing) it.

    python -m analysis.fork_features --gens results/rl_delta/olmo2-1b-sft_train300_n16
"""

import argparse
import json
from collections import Counter

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import prompts
from common.io import read_jsonl, write_json
from common.models import resolve


def auc(score, label):
    score, label = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inv, weights=ranks) / cnt)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="olmo2-1b-sft")
    ap.add_argument("--label", choices=["majority", "gold"], default="gold")
    ap.add_argument("--max_problems", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    hf_id, fmt = resolve(args.model)
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.bfloat16).cuda().eval()
    L = model.config.num_hidden_layers
    norm, head = model.model.norm, model.lm_head
    rows = read_jsonl(f"{args.gens}/{args.split}.generations.jsonl")[: args.max_problems]

    # pass 1: entropies to fix the fork threshold (cheap: no hidden states)
    feats = []
    H_pool = []
    cache = []
    for r in rows:
        prompt = prompts.build_prompt(fmt, r["question"], [], tok)
        p_ids = tok(prompt, add_special_tokens=(fmt == "plain")).input_ids
        answers = [s["lenient_answer"] for s in r["scores"]]
        maj = Counter(a for a in answers if a is not None).most_common(1)
        maj = maj[0][0] if maj else None
        for out, s in zip(r["outputs"], r["scores"]):
            good = s["lenient_correct"] if args.label == "gold" else (s["lenient_answer"] is not None and s["lenient_answer"] == maj)
            o_ids = tok(out, add_special_tokens=False).input_ids[:512]
            if len(o_ids) >= 2:
                cache.append((p_ids, o_ids, bool(good)))
    print(f"{len(cache)} samples", flush=True)

    for n, (p_ids, o_ids, good) in enumerate(cache):
        ids = torch.tensor([p_ids + o_ids], device="cuda")
        P, T = len(p_ids), len(o_ids)
        with torch.no_grad():
            res = model(ids, output_hidden_states=True)
            lp = torch.log_softmax(res.logits[0, P - 1:P - 1 + T].float(), -1)       # predicts o_ids[t]
            p = lp.exp()
            H = -(p * lp).sum(-1)
            y = torch.tensor(o_ids, device="cuda")
            lp_y = lp.gather(-1, y[:, None]).squeeze(-1)
            rank = (lp > lp_y[:, None]).sum(-1) + 1
            gap = lp.max(-1).values - lp_y
            top2 = p.topk(2, -1).values
            margin = top2[:, 0] - top2[:, 1]
            # logit-lens argmax of every layer at every position (positions predicting o_ids[t])
            lens = torch.stack([head(norm(res.hidden_states[l][0, P - 1:P - 1 + T])).argmax(-1)
                                for l in range(1, L)] + [lp.argmax(-1)])                # L x T
            match = (lens == y[None, :])                                                 # L x T
            # settle depth: first layer from which every later layer's argmax is y
            suffix_all = torch.flip(torch.cumprod(torch.flip(match.int(), [0]), 0), [0]).bool()
            settle = torch.where(suffix_all.any(0), suffix_all.int().argmax(0) + 1, torch.full_like(rank, L + 1))
            late = match[-4:].float().mean(0)
        H, rank, gap, margin, settle, late = (x.cpu().numpy() for x in (H, rank, gap, margin, settle, late))
        H_pool.extend(H.tolist())
        for t in range(T - 1):
            feats.append((good, float(H[t]), int(rank[t]), float(gap[t]), int(settle[t]), float(late[t]),
                          float(H[t + 1]), float(margin[t + 1])))
        if (n + 1) % 500 == 0:
            print(f"{n + 1}/{len(cache)}", flush=True)

    thr = float(np.quantile(np.array(H_pool), 0.8))
    F = np.array([f for f in feats if f[1] > thr], dtype=float)
    good = F[:, 0].astype(bool)
    names = ["rank", "logp_gap", "settle_depth", "late_agree", "next_entropy", "next_margin"]
    cols = {nm: F[:, i + 2] for i, nm in enumerate(names)}
    nongreedy = cols["rank"] >= 2
    res = {"model": args.model, "label": args.label, "fork_thr": thr, "fork_positions": int(len(F)),
           "good_share": float(good.mean()), "nongreedy_share": float(nongreedy.mean()),
           "auroc_all_forks": {nm: auc(v, good) for nm, v in cols.items()},
           "auroc_nongreedy_forks": {nm: auc(v[nongreedy], good[nongreedy]) for nm, v in cols.items()},
           "means_good_vs_bad": {nm: [float(v[good].mean()), float(v[~good].mean())] for nm, v in cols.items()}}
    out = args.out or f"{args.gens}/fork_features_{args.label}.json"
    write_json(out, res)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
