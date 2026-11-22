"""Token-level signals on saved generations: entropy / fork tokens and deep-thinking ratio.

compute: re-runs each saved (prompt, output) through the model once (teacher forcing) and
         stores, for every generated token,
           - entropy of the final-layer next-token distribution (nats)    -> fork tokens
             (Wang et al. 2025: the ~20% highest-entropy tokens act as "forks")
           - JSD(p_L || p_l) for every layer l = 1..L (nats, logit lens)  -> DTR
             (Chen et al. 2026, arXiv 2602.13517)
         Everything threshold-dependent is applied later in `report`, so g / rho / the fork
         quantile can be changed without another GPU pass.
report:  per-run summary + correct-vs-incorrect comparison (AUROC) for DTR, fork share,
         mean entropy and length.

    python -m analysis.token_signals compute --run results/olmo2-1b-sft/gsm8k/0shot
    python -m analysis.token_signals report results/olmo2-1b-*/gsm8k/*shot

Logit-lens details (choices the DTR paper leaves open):
  - intermediate layer l < L: lm_head(final_norm(h_l)), the usual logit lens. HF's last
    hidden state is already normed, so layer L is lm_head(hidden_states[-1]) = the logits.
  - JSD in nats (max ln 2 ~ 0.693); paper default g = 0.5 is applied in nats unless
    --log_base 2 is given (then D is divided by ln 2 before thresholding).
  - deep-thinking layers follow the paper's Eq. 5: l >= ceil(rho * L) (not Algorithm 1).
  - Only the saved (truncated) output is scored, re-tokenized after the prompt; EOS excluded.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

from common.io import read_jsonl, write_json

LN2 = math.log(2.0)


# ---------------------------------------------------------------- compute

def _entropy(logp):
    return -(logp.exp() * logp).sum(-1)


def compute(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from common import prompts
    from common.models import resolve

    run = Path(args.run)
    meta = json.loads((run / f"{args.split}.metrics.json").read_text())
    gens = read_jsonl(run / f"{args.split}.generations.jsonl")[: args.limit]
    hf_id, fmt = resolve(meta["model"], meta.get("format"))
    data_dir = args.data_dir or f"data/{meta['task']}"

    tok = AutoTokenizer.from_pretrained(hf_id)
    exemplars = []
    if meta.get("few_shot"):
        exemplars = random.Random(meta.get("seed", 0)).sample(read_jsonl(f"{data_dir}/train.jsonl"),
                                                              meta["few_shot"])
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.bfloat16).cuda().eval()
    norm, head = model.model.norm, model.lm_head
    L = model.config.num_hidden_layers

    ent, jsd, tokid, offsets, per = [], [], [], [0], []
    checked = False
    for i, g in enumerate(gens):
        prompt = prompts.build_prompt(fmt, g["question"], exemplars, tok)
        out = g["outputs"][0]
        p_ids = tok(prompt, add_special_tokens=(fmt == "plain")).input_ids
        o_ids = tok((" " + out) if fmt == "plain" else out, add_special_tokens=False).input_ids
        o_ids = o_ids[: args.max_tokens]
        if not o_ids:
            o_ids = []
        ids = torch.tensor([p_ids + o_ids], device="cuda")
        T, P = len(o_ids), len(p_ids)
        if T:
            with torch.no_grad():
                res = model(ids, output_hidden_states=True)
                hs = res.hidden_states  # (emb, h_1, ..., h_{L-1}, norm(h_L))
                pos = slice(P - 1, P - 1 + T)  # positions that predict the output tokens
                final = torch.log_softmax(head(hs[-1][0, pos]).float(), -1)
                if not checked:
                    ref = torch.log_softmax(res.logits[0, pos].float(), -1)
                    assert torch.allclose(final, ref, atol=1e-2), "last hidden state is not the logits input"
                    checked = True
                pf = final.exp()
                H = _entropy(final)
                D = torch.empty(T, L, device="cuda")
                for l in range(1, L):
                    lp = torch.log_softmax(head(norm(hs[l][0, pos])).float(), -1)
                    m = 0.5 * (pf + lp.exp())
                    D[:, l - 1] = _entropy(m.clamp_min(1e-30).log()) - 0.5 * H - 0.5 * _entropy(lp)
                D[:, L - 1] = 0.0
            ent.append(H.cpu().numpy())
            jsd.append(D.clamp_min(0).half().cpu().numpy())
            tokid.append(np.array(o_ids, dtype=np.int32))
        offsets.append(offsets[-1] + T)
        s = g["scores"][0]
        per.append({"lenient_correct": bool(s["lenient_correct"]), "strict_correct": bool(s["strict_correct"]),
                    "truncated": bool(s.get("truncated", False)), "n_tokens": T})
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(gens)}", flush=True)

    out = run / f"{args.split}.token_signals.npz"
    np.savez_compressed(
        out, entropy=np.concatenate(ent) if ent else np.zeros(0, np.float32),
        jsd=np.concatenate(jsd) if jsd else np.zeros((0, L), np.float16),
        token_ids=np.concatenate(tokid) if tokid else np.zeros(0, np.int32),
        offsets=np.array(offsets), lenient_correct=np.array([p["lenient_correct"] for p in per]),
        strict_correct=np.array([p["strict_correct"] for p in per]),
        truncated=np.array([p["truncated"] for p in per]), num_layers=L, hf_id=hf_id)
    print(f"saved {out}  ({offsets[-1]} tokens, {len(per)} samples, L={L})")


# ---------------------------------------------------------------- report

def settling_depth(jsd, g):
    """c_t = first layer (1-based) where the running min of D_{t,l} is <= g (Eqs. 3-4)."""
    run_min = np.minimum.accumulate(jsd.astype(np.float32), axis=1)
    return (run_min <= g).argmax(axis=1) + 1  # D_{t,L} = 0, so always defined


def auroc(score, label):
    """P(score of a correct sample > score of an incorrect one); 0.5 = no signal."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]), kind="mergesort"), kind="mergesort") + 1.0
    # average ranks for ties
    allv = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def report_one(path, g, rho, log_base, fork_q, fork_thr, prefix):
    z = np.load(path)
    L = int(z["num_layers"])
    off, ent, jsd = z["offsets"], z["entropy"], z["jsd"].astype(np.float32)
    if log_base == 2:
        jsd = jsd / LN2
    deep_from = math.ceil(rho * L)
    c = settling_depth(jsd, g) if len(jsd) else np.zeros(0, int)
    deep = c >= deep_from
    thr = fork_thr if fork_thr is not None else (float(np.quantile(ent, fork_q)) if len(ent) else float("nan"))
    fork = ent > thr

    rows = []
    for i in range(len(off) - 1):
        a, b = off[i], off[i + 1]
        bp = min(b, a + prefix) if prefix else b
        if b == a:
            rows.append((0, np.nan, np.nan, np.nan))
            continue
        rows.append((b - a, deep[a:bp].mean(), fork[a:b].mean(), ent[a:b].mean()))
    rows = np.array(rows, dtype=float)
    ok = ~np.isnan(rows[:, 1])
    y = z["lenient_correct"][ok]
    r = rows[ok]
    res = {
        "samples": int(ok.sum()), "acc_lenient": float(y.mean()), "tokens": int(off[-1]), "layers": L,
        "fork_threshold_nats": thr, "deep_from_layer": deep_from,
        "mean_settling_depth": float(c.mean()) if len(c) else None,
        "DTR_mean": float(r[:, 1].mean()), "fork_share_mean": float(r[:, 2].mean()),
        "entropy_mean": float(r[:, 3].mean()), "length_mean": float(r[:, 0].mean()),
        "DTR_correct": float(r[y, 1].mean()) if y.any() else None,
        "DTR_incorrect": float(r[~y, 1].mean()) if (~y).any() else None,
        "AUROC_DTR": auroc(r[:, 1], y), "AUROC_fork_share": auroc(r[:, 2], y),
        "AUROC_entropy": auroc(r[:, 3], y), "AUROC_length": auroc(r[:, 0], y),
    }
    return res, z, fork, off


def top_fork_tokens(z, fork, tok, k=25):
    from collections import Counter
    ids = z["token_ids"][fork]
    cnt = Counter(ids.tolist())
    tot = Counter(z["token_ids"].tolist())
    return [(tok.decode([t]), n, round(n / tot[t], 2)) for t, n in cnt.most_common(k)]


def report(args):
    rows = []
    for run in args.runs:
        p = Path(run) / f"{args.split}.token_signals.npz"
        if not p.exists():
            print(f"missing {p}", file=sys.stderr)
            continue
        res, z, fork, _ = report_one(p, args.g, args.rho, args.log_base, args.fork_q, args.fork_thr, args.prefix)
        res["run"] = str(Path(run))
        rows.append(res)
        write_json(Path(run) / f"{args.split}.token_signals.summary.json",
                   {**res, "g": args.g, "rho": args.rho, "log_base": args.log_base, "prefix": args.prefix})
        if args.show_tokens:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(str(z["hf_id"]))
            print(f"\n{run}: most frequent fork tokens (token, count, share of that token's uses that are forks)")
            print(top_fork_tokens(z, fork, tok))
    if not rows:
        return
    import pandas as pd
    cols = ["run", "samples", "acc_lenient", "length_mean", "entropy_mean", "fork_threshold_nats",
            "fork_share_mean", "mean_settling_depth", "DTR_mean", "DTR_correct", "DTR_incorrect",
            "AUROC_DTR", "AUROC_fork_share", "AUROC_entropy", "AUROC_length"]
    df = pd.DataFrame(rows)[cols]
    print(f"\ng={args.g} rho={args.rho} (deep = settles at layer >= ceil(rho*L)) log_base={args.log_base} "
          f"fork={'q' + str(args.fork_q) if args.fork_thr is None else args.fork_thr} prefix={args.prefix or 'all'}")
    print(df.to_string(index=False, float_format="%.3f"))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compute")
    c.add_argument("--run", required=True, help="results dir containing <split>.generations.jsonl")
    c.add_argument("--split", default="test")
    c.add_argument("--data_dir", default=None)
    c.add_argument("--limit", type=int, default=None)
    c.add_argument("--max_tokens", type=int, default=1024)
    r = sub.add_parser("report")
    r.add_argument("runs", nargs="+")
    r.add_argument("--split", default="test")
    r.add_argument("--g", type=float, default=0.5)
    r.add_argument("--rho", type=float, default=0.85)
    r.add_argument("--log_base", type=float, default=math.e, choices=[math.e, 2])
    r.add_argument("--fork_q", type=float, default=0.8, help="per-run entropy quantile for fork tokens")
    r.add_argument("--fork_thr", type=float, default=None, help="fixed entropy threshold (nats) instead")
    r.add_argument("--prefix", type=int, default=0, help="DTR over the first N tokens only (Think@n uses 50)")
    r.add_argument("--show_tokens", action="store_true")
    args = ap.parse_args()
    compute(args) if args.cmd == "compute" else report(args)


if __name__ == "__main__":
    main()
