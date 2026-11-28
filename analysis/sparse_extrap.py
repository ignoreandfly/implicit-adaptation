"""Sparse weight extrapolation along the RL step (DPO -> Instruct), OLMo 2 1B.

Dense extrapolation theta = A + alpha * (B - A) with A = DPO, B = Instruct (RLVR) and alpha = 1.5 beat Instruct
on the full test (69.9 vs 67.6, p = 0.020; README). RL is reported to update only a small subnetwork
(Mukherjee et al. 2025, arXiv 2505.11711). Question: is the extrapolation gain carried by a small fraction of the
weights?

    theta = A + alpha * (M * Delta),   Delta = B - A,  M = mask keeping the top-q fraction of |Delta|
            (per weight matrix), or a random q-fraction of entries (control).

Step 1 (stats, CPU): fraction of entries RL changed at all, and how concentrated |Delta|^2 is in the top-q entries.
Step 2 (eval): every config is built on CPU, saved (bf16) to a scratch dir and evaluated greedily with
evaluation.generate --backend vllm on the full GSM8K test, together with the dense alpha = 1 (= Instruct) and
alpha = 1.5 references in the same pipeline.

    python -m analysis.sparse_extrap --configs dense:1 dense:1.5 top:0.05:1.5 rand:0.05:1.5 --out results/sparse_extrap
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.models import resolve


def load_sd(name):
    m = AutoModelForCausalLM.from_pretrained(resolve(name)[0], torch_dtype=torch.float32)
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def topq_mask(d, q):
    """Keep the q-fraction of entries of d with the largest |d| (per tensor; ties may keep a few more)."""
    if q >= 1:
        return torch.ones_like(d, dtype=torch.bool)
    k = max(1, int(round(q * d.numel())))
    thr = d.abs().flatten().kthvalue(d.numel() - k + 1).values
    return d.abs() >= thr


def stats(delta, qs):
    tot = sum(v.numel() for v in delta.values())
    changed = sum(int((v != 0).sum()) for v in delta.values())
    energy = sum(float((v.double() ** 2).sum()) for v in delta.values())
    out = {"params": tot, "frac_changed": changed / tot, "energy_in_top": {}}
    for q in qs:
        e = 0.0
        for v in delta.values():
            e += float((v[topq_mask(v, q)].double() ** 2).sum())
        out["energy_in_top"][str(q)] = e / energy
    by = {}
    for k, v in delta.items():
        kind = next((t for t in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                                 "embed_tokens", "lm_head", "norm"] if t in k), "other")
        a = by.setdefault(kind, [0, 0, 0.0])
        a[0] += v.numel(); a[1] += int((v != 0).sum()); a[2] += float((v.double() ** 2).sum())
    out["by_module"] = {k: {"frac_changed": a[1] / a[0], "share_of_energy": a[2] / energy} for k, a in by.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="olmo2-1b-dpo")
    ap.add_argument("--b", default="olmo2-1b-instruct")
    ap.add_argument("--configs", nargs="+", required=True,
                    help="dense:ALPHA | top:Q:ALPHA | rand:Q:ALPHA")
    ap.add_argument("--subset", type=int, default=None, help="evaluate on a fixed test subset (default: full test)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--scratch", default="checkpoints/sparse_extrap_tmp")
    ap.add_argument("--mem", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/sparse_extrap")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    sd_a, sd_b = load_sd(args.a), load_sd(args.b)
    delta = {k: sd_b[k] - sd_a[k] for k in sd_a if sd_a[k].is_floating_point()}
    del sd_b                                    # RAM: keep only A and Delta (+ the model being written)
    if not (out / "stats.json").exists():
        st = stats(delta, [0.01, 0.05, 0.2, 0.5])
        json.dump(st, open(out / "stats.json", "w"), indent=2)
        print("stats:", json.dumps({k: st[k] for k in ["frac_changed", "energy_in_top"]}), flush=True)

    tok = AutoTokenizer.from_pretrained(resolve(args.b)[0])
    model = AutoModelForCausalLM.from_pretrained(resolve(args.b)[0], torch_dtype=torch.float32)
    model.config.torch_dtype = torch.bfloat16    # saved fp32; vLLM loads it as bf16 (--dtype auto -> bf16 on L4)
    g = torch.Generator().manual_seed(args.seed)
    summary_p = out / f"summary_{args.split}.json"
    summary = json.load(open(summary_p)) if summary_p.exists() else {}
    for cfg in args.configs:
        name = cfg.replace(":", "_")
        if "acc" in summary.get(name, {}):
            print(f"{cfg}: done {summary[name]}", flush=True); continue
        parts = cfg.split(":")
        kind, alpha = parts[0], float(parts[-1])
        q = float(parts[1]) if kind != "dense" else 1.0
        kept = 0
        target = model.state_dict()            # tensors share storage with the model: write in place
        for k, v in sd_a.items():
            if k not in delta:
                continue                          # non-float buffers: keep B's
            d = delta[k]
            if kind == "dense":
                m = None
            elif kind == "top":
                m = topq_mask(d, q)
            elif kind == "rand":
                m = torch.rand(d.shape, generator=g) < q
            else:
                raise ValueError(cfg)
            with torch.no_grad():
                target[k].copy_(v + alpha * (d if m is None else d * m))
            kept += d.numel() if m is None else int(m.sum())
        os.makedirs(args.scratch, exist_ok=True)
        model.save_pretrained(args.scratch, safe_serialization=True, max_shard_size="20GB")
        tok.save_pretrained(args.scratch)
        ev = out / "eval" / name
        cmd = [sys.executable, "-m", "evaluation.generate", "--backend", "vllm", "--model", args.scratch,
               "--format", "chat", "--split", args.split, "--gpu_memory_utilization", str(args.mem), "--out", str(ev)]
        if args.subset:
            cmd += ["--subset", str(args.subset)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        try:
            m = json.load(open(ev / f"{args.split}.metrics.json"))
            summary[name] = {"acc": m["acc_lenient"], "frac_kept": kept / sum(d.numel() for d in delta.values()),
                             "trunc": m.get("truncation_rate")}
        except Exception:
            summary[name] = {"error": r.stderr[-600:]}
        print(f"{cfg}: {summary[name]}", flush=True)
        json.dump(summary, open(summary_p, "w"), indent=2)


if __name__ == "__main__":
    main()
