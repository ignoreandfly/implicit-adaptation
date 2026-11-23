"""Which decision-point criterion predicts where post-training actually changes the model?

Teacher-force model A's own greedy GSM8K solutions (training questions) through A and through a later checkpoint B.
At every response position: A's next-token entropy H_A (ReasonMaxxer's gate), A's top-2 probability margin m_A (our
fork gate), and whether B's greedy token differs from A's ("changed decision"). Reports, for each criterion:
  - AUROC for predicting changed decisions,
  - recall / precision of the top-s% positions (s = 1, 5, 10, 20), i.e. the share of B's changed decisions that a
    gate of that sparsity would train on,
  - overlap (Jaccard) of the two gates' top-5% sets, and how many changes fall in low-entropy near-ties
    (H_A <= 1.4 nats but m_A <= 0.1), i.e. binary forks an entropy gate misses.

    python -m analysis.gate_vs_rl --gens results/rl_delta/olmo2-1b-dpo_train300 --a olmo2-1b-dpo --b olmo2-1b-instruct
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import prompts
from common.io import read_jsonl
from common.models import resolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(resolve(args.b)[0])
    ma = AutoModelForCausalLM.from_pretrained(resolve(args.a)[0], torch_dtype=torch.bfloat16).cuda().eval()
    mb = AutoModelForCausalLM.from_pretrained(resolve(args.b)[0], torch_dtype=torch.bfloat16).cuda().eval()
    rows = read_jsonl(f"{args.gens}/{args.split}.generations.jsonl")
    H, M, C, Q = [], [], [], []
    with torch.no_grad():
        for qi, r in enumerate(tqdm(rows, desc="teacher-force", mininterval=20)):
            p_ids = tok(prompts.build_prompt("chat", r["question"], [], tok), add_special_tokens=False).input_ids
            o_ids = tok(r["outputs"][0], add_special_tokens=False).input_ids[: args.max_tokens]
            if not o_ids:
                continue
            ids = torch.tensor([p_ids + o_ids], device="cuda")
            P = len(p_ids)
            la = torch.log_softmax(ma(ids).logits[0, P - 1:-1].float(), -1)
            lb = mb(ids).logits[0, P - 1:-1].float()
            pa = la.exp()
            top = pa.topk(2, -1).values
            H.append((-(pa * la).sum(-1)).cpu().numpy())
            M.append((top[:, 0] - top[:, 1]).cpu().numpy())
            C.append((la.argmax(-1) != lb.argmax(-1)).cpu().numpy())
            Q.append(np.full(len(o_ids), qi))
    H, M, C = np.concatenate(H), np.concatenate(M), np.concatenate(C).astype(bool)
    n, nc = len(C), int(C.sum())
    res = {"a": args.a, "b": args.b, "positions": n, "changed": nc, "changed_share": nc / n,
           "auroc": {"entropy": float(roc_auc_score(C, H)), "neg_margin": float(roc_auc_score(C, -M))}, "gates": {}}
    for s in (0.01, 0.05, 0.10, 0.20):
        k = int(round(s * n))
        for name, score in (("entropy", H), ("margin", -M)):
            sel = np.zeros(n, bool); sel[np.argsort(-score)[:k]] = True
            res["gates"][f"{name}_{int(s * 100)}pct"] = {"recall": float((sel & C).sum() / max(1, nc)),
                                                       "precision": float((sel & C).sum() / max(1, k))}
    k = int(round(0.05 * n))
    se = set(np.argsort(-H)[:k]); sm = set(np.argsort(M)[:k])
    res["jaccard_top5pct"] = len(se & sm) / len(se | sm)
    low_ent_ties = (H <= 1.4) & (M <= 0.1)
    res["low_entropy_near_ties"] = {"share_of_positions": float(low_ent_ties.mean()),
                                    "share_of_changes": float((low_ent_ties & C).sum() / max(1, nc)),
                                    "changes_with_H_gt_1.4": float(((H > 1.4) & C).sum() / max(1, nc))}
    out = Path(args.out or f"results/gate_vs_rl/{Path(args.a).name}_vs_{Path(args.b).name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
