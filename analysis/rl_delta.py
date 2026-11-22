"""Where does a later post-training stage change the model? (step 1 of the "sparse RL rules" plan)

Teacher-force the SAME trajectories (default: the earlier stage's own greedy GSM8K solutions)
through two OLMo 2 checkpoints A (e.g. SFT) and B (e.g. Instruct = SFT+DPO+RLVR), and compare
their next-token distributions at every response position (cf. "Sparse but Critical",
arXiv 2603.22446; "On the Direction of RLVR Updates", arXiv 2603.22117):

  JSD(p_A || p_B)             how much the distribution moves (nats)
  argmax disagreement         would B pick a different greedy token here?
  dlogp = log p_B(y) - log p_A(y) for the token y actually in the trajectory (signed)
  H_A                         entropy of A (is the change at a fork token?)
  position class              first response token / sentence start / other

Aggregates: sparsity (share of positions with JSD above thresholds, share of disagreements),
where disagreements happen (first token, sentence starts, forks), and which tokens B promotes
or demotes overall (summed probability-mass shift p_B - p_A, restricted to tokens in either
model's top-10 at that position), with example contexts. Saves a per-disagreement table.

    python -m analysis.rl_delta --gens results/rl_delta/sft_train300 --a olmo2-1b-sft --b olmo2-1b-instruct
"""

import argparse
import json
import math
from collections import Counter, defaultdict

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import prompts
from common.io import read_jsonl, write_json, write_jsonl
from common.models import resolve

SENT_END = (".", "!", "?", "\n", ":")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", required=True, help="run dir with <split>.generations.jsonl (trajectories)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--a", default="olmo2-1b-sft")
    ap.add_argument("--b", default="olmo2-1b-instruct")
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    hf_a, fmt = resolve(args.a)
    hf_b, fmt_b = resolve(args.b)
    assert fmt == fmt_b == "chat", "both checkpoints should be chat-format post-trained models"
    tok = AutoTokenizer.from_pretrained(hf_a)
    A = AutoModelForCausalLM.from_pretrained(hf_a, torch_dtype=torch.bfloat16).cuda().eval()
    B = AutoModelForCausalLM.from_pretrained(hf_b, torch_dtype=torch.bfloat16).cuda().eval()
    rows = read_jsonl(f"{args.gens}/{args.split}.generations.jsonl")

    jsd_all, H_all, poscls_all, disagree_all = [], [], [], []
    mass_shift = defaultdict(float)          # token id -> summed (p_B - p_A) over positions
    mass_count = Counter()
    dis_rows = []
    for i, r in enumerate(rows):
        prompt = prompts.build_prompt("chat", r["question"], [], tok)
        p_ids = tok(prompt, add_special_tokens=False).input_ids
        o_ids = tok(r["outputs"][0], add_special_tokens=False).input_ids[: args.max_tokens]
        if not o_ids:
            continue
        ids = torch.tensor([p_ids + o_ids], device="cuda")
        sl = slice(len(p_ids) - 1, len(p_ids) - 1 + len(o_ids))
        with torch.no_grad():
            la = torch.log_softmax(A(ids).logits[0, sl].float(), -1)
            lb = torch.log_softmax(B(ids).logits[0, sl].float(), -1)
        pa, pb = la.exp(), lb.exp()
        m = (0.5 * (pa + pb)).clamp_min(1e-30).log()
        jsd = (0.5 * (pa * (la - m)).sum(-1) + 0.5 * (pb * (lb - m)).sum(-1)).clamp_min(0)
        H = -(pa * la).sum(-1)
        y = torch.tensor(o_ids, device="cuda")
        dlogp = (lb.gather(-1, y[:, None]) - la.gather(-1, y[:, None])).squeeze(-1)
        arg_a, arg_b = la.argmax(-1), lb.argmax(-1)
        # top-10 union mass shift
        top = torch.cat([la.topk(10, -1).indices, lb.topk(10, -1).indices], -1)
        shift = (pb - pa).gather(-1, top)
        for t in range(len(o_ids)):
            for tid, s in zip(top[t].tolist(), shift[t].tolist()):
                mass_shift[tid] += s / 2  # each id may appear twice in the union; halve
                mass_count[tid] += 1
        text_before = [tok.decode(o_ids[:t]) for t in range(len(o_ids))]
        for t in range(len(o_ids)):
            prev = text_before[t].rstrip(" ")
            cls = "first" if t == 0 else ("sentence_start" if prev.endswith(SENT_END) else "other")
            dis = bool(arg_a[t] != arg_b[t])
            jsd_all.append(float(jsd[t])); H_all.append(float(H[t])); poscls_all.append(cls); disagree_all.append(dis)
            if dis:
                dis_rows.append({"problem": i, "t": t, "cls": cls, "H_A": float(H[t]), "jsd": float(jsd[t]),
                                 "a_tok": tok.decode([int(arg_a[t])]), "b_tok": tok.decode([int(arg_b[t])]),
                                 "p_a_of_b_tok": float(pa[t, arg_b[t]]), "rank_b_tok_under_a": int((la[t] > la[t, arg_b[t]]).sum()) + 1,
                                 "context": text_before[t][-80:]})
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(rows)}", flush=True)

    jsd_all, H_all, dis = np.array(jsd_all), np.array(H_all), np.array(disagree_all)
    cls = np.array(poscls_all)
    fork_thr = float(np.quantile(H_all, 0.8))
    by_cls = {c: {"positions": int((cls == c).sum()), "disagree_rate": float(dis[cls == c].mean()),
                  "mean_jsd": float(jsd_all[cls == c].mean())} for c in ("first", "sentence_start", "other")}
    promoted = sorted(mass_shift.items(), key=lambda kv: -kv[1])[:40]
    demoted = sorted(mass_shift.items(), key=lambda kv: kv[1])[:40]
    swaps = Counter(f'{d["a_tok"]!r} -> {d["b_tok"]!r}' for d in dis_rows).most_common(40)
    res = {
        "a": args.a, "b": args.b, "trajectories": args.gens, "problems": len(rows), "positions": int(len(jsd_all)),
        "jsd_quantiles": {q: float(np.quantile(jsd_all, q)) for q in (0.5, 0.8, 0.9, 0.95, 0.99)},
        "share_jsd_gt": {thr: float((jsd_all > thr).mean()) for thr in (0.01, 0.05, 0.1, 0.2)},
        "disagree_rate": float(dis.mean()),
        "share_of_total_jsd_in_top5pct_positions": float(np.sort(jsd_all)[-max(1, len(jsd_all) // 20):].sum() / jsd_all.sum()),
        "fork_threshold_H_A": fork_thr,
        "disagreements_at_forks": float((H_all[dis] > fork_thr).mean()) if dis.any() else None,
        "disagree_rate_at_forks": float(dis[H_all > fork_thr].mean()),
        "disagree_rate_non_forks": float(dis[H_all <= fork_thr].mean()),
        "by_position_class": by_cls,
        "disagreement_rank_of_b_token_under_a": {k: float(np.mean([d["rank_b_tok_under_a"] == k for d in dis_rows])) for k in (2, 3)}
                                               | {"4+": float(np.mean([d["rank_b_tok_under_a"] >= 4 for d in dis_rows]))},
        "top_promoted_tokens": [(tok.decode([t]), round(v, 3), mass_count[t]) for t, v in promoted],
        "top_demoted_tokens": [(tok.decode([t]), round(v, 3), mass_count[t]) for t, v in demoted],
        "top_argmax_swaps_a_to_b": swaps,
    }
    out = args.out or f"results/rl_delta/{args.a}_vs_{args.b}"
    write_json(f"{out}/summary.json", res)
    write_jsonl(f"{out}/disagreements.jsonl", dis_rows)
    print(json.dumps({k: v for k, v in res.items() if not k.startswith("top_")}, indent=1))
    print("promoted:", res["top_promoted_tokens"][:20])
    print("demoted:", res["top_demoted_tokens"][:20])
    print("swaps:", swaps[:20])


if __name__ == "__main__":
    main()
