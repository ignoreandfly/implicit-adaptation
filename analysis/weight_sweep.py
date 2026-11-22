"""Training-free weight interpolation / extrapolation along the OLMo 2 post-training ladder.

For checkpoints A and B of the same architecture: theta(alpha) = theta_A + alpha * (theta_B - theta_A).
alpha = 0 -> A, alpha = 1 -> B, alpha > 1 extrapolates past B (cf. ExPO-style extrapolation for alignment).
Each theta(alpha) is built on CPU in fp32, saved (bf16) to a scratch directory that is overwritten for the next
alpha, and evaluated greedily with methods.run (rl_rules --rl_beta 0, chat format), i.e. the same decoder as the
other SFT-family numbers (dev: SFT 0.720, Instruct 0.790).

    python -m analysis.weight_sweep --a olmo2-1b-sft --b olmo2-1b-dpo --alphas 0.5 1.5 2 3 --split dev --gpu 0
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.io import write_json
from common.models import resolve


def load_sd(name):
    path = resolve(name)[0]
    m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
    return path, {k: v.detach().clone() for k, v in m.state_dict().items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--alphas", type=float, nargs="+", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--scratch", default="/data/azfarm/siddhant_fork_lora/weight_sweep_tmp")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()

    tag = args.tag or f"{Path(args.a).name}__{Path(args.b).name}"
    out_root = Path(f"results/weight_sweep/{tag}")
    path_a, sd_a = load_sd(args.a)
    path_b, sd_b = load_sd(args.b)
    assert sd_a.keys() == sd_b.keys(), "checkpoints have different parameter names"
    for k in sd_a:
        assert sd_a[k].shape == sd_b[k].shape, f"shape mismatch at {k}"
    tok = AutoTokenizer.from_pretrained(path_b)
    model = AutoModelForCausalLM.from_pretrained(path_b, torch_dtype=torch.float32)
    results = {}
    for alpha in args.alphas:
        sd = {k: sd_a[k] + alpha * (sd_b[k] - sd_a[k]) for k in sd_a}
        model.load_state_dict(sd)
        os.makedirs(args.scratch, exist_ok=True)
        model.to(torch.bfloat16).save_pretrained(args.scratch, safe_serialization=True)
        model.to(torch.float32)
        tok.save_pretrained(args.scratch)
        out = out_root / f"alpha_{alpha:g}"
        cmd = [sys.executable, "-m", "methods.run", "--method", "rl_rules", "--rl_beta", "0", "--model", args.scratch,
               "--format", "chat", "--split", args.split, "--max_new_tokens", str(args.max_new_tokens),
               "--batch_problems", "2", "--out", str(out)]
        log = subprocess.run(cmd, env={**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu}, capture_output=True, text=True)
        try:
            m = json.load(open(out / f"{args.split}.metrics.json"))
            results[alpha] = {"acc": m["acc_lenient"], "trunc": m.get("truncation_rate"), "tokens": m.get("mean_gen_tokens")}
        except Exception:
            results[alpha] = {"error": log.stderr[-400:]}
        print(f"alpha={alpha:g}: {results[alpha]}", flush=True)
    write_json(out_root / f"summary_{args.split}.json", {"a": args.a, "b": args.b, "split": args.split, "results": results})


if __name__ == "__main__":
    main()
