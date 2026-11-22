"""Screening harness for training-free decoding methods (one model, dev split, fast).

    python -m methods.run --method cot_decoding --model olmo2-1b-base --split dev --k 10

Writes results/screen/<model>/<split>/<method>_<tag>/{<split>.generations.jsonl, <split>.metrics.json}.
The metrics file has the standard fields (from evaluation.metrics.aggregate, computed on the
method's chosen output) plus method-specific fields from <method>.summarize().
"""

import argparse
import importlib
import json
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import prompts
from common.io import git_hash, read_jsonl, write_json, write_jsonl
from common.models import resolve
from evaluation.generate import stop_ids
from evaluation.metrics import aggregate
from tasks import get_task

METHODS = ["cot_decoding", "greedy", "dola", "budget_forcing", "sample", "cfg", "contrastive", "eager", "tlsc", "power_smc", "rule_adapter", "rl_rules", "fork_mbr"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=METHODS, required=True)
    ap.add_argument("--model", default="olmo2-1b-base")
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--few_shot", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--batch_problems", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default=None)
    # cot_decoding
    ap.add_argument("--k", type=int, default=10)
    # greedy / budget_forcing: text forced after "Answer:" (e.g. " Let's think step by step.")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--rep_penalty", type=float, default=1.0)
    # dola
    ap.add_argument("--dola_layers", choices=["low", "high"], default="high")
    ap.add_argument("--dola_alpha", type=float, default=0.1)
    # budget_forcing
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--wait_text", default=" Wait, let me check that again.")
    # sample
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--fork_threshold", type=float, default=None)
    ap.add_argument("--t_low", type=float, default=0.3)
    ap.add_argument("--t_high", type=float, default=1.0)
    # cfg
    ap.add_argument("--cfg_gamma", type=float, default=1.5)
    # contrastive (dropout amateur)
    ap.add_argument("--cd_beta", type=float, default=0.5)
    ap.add_argument("--cd_alpha", type=float, default=0.1)
    ap.add_argument("--dropout", type=float, default=0.2)
    # eager (uses --n as the per-prompt sequence budget M)
    ap.add_argument("--eager_theta", type=float, default=2.0)
    ap.add_argument("--eager_tau", type=float, default=0.6)
    # tlsc: thought-level self-certainty (uses --temperature for candidates)
    ap.add_argument("--tlsc_k", type=int, default=4)
    ap.add_argument("--tlsc_steps", type=int, default=40)
    # power_smc (uses --n as the number of particles N)
    ap.add_argument("--smc_alpha", type=float, default=4.0)
    ap.add_argument("--smc_kappa", type=float, default=0.5)
    ap.add_argument("--smc_ramp", type=int, default=100)
    # rule_adapter (Hewitt-style rules; see methods/rule_adapter.py)
    ap.add_argument("--ra_stop_pen", type=float, default=5.0)
    ap.add_argument("--ra_end_bonus", type=float, default=5.0)
    ap.add_argument("--ra_min_len", type=int, default=-1, help="stop penalty only before this many tokens (-1: until answered)")
    ap.add_argument("--ra_step_bonus", type=float, default=1.0)
    ap.add_argument("--ra_drift_pen", type=float, default=4.0)
    ap.add_argument("--ra_first_digit_pen", type=float, default=5.0)
    ap.add_argument("--ra_first_step_bonus", type=float, default=3.0)
    ap.add_argument("--ra_rep_pen", type=float, default=1.5)
    ap.add_argument("--ra_ngram_pen", type=float, default=5.0)
    ap.add_argument("--ra_n", type=int, default=1, help="samples per question (>1: sample + majority vote)")
    ap.add_argument("--ra_off", default="", help="comma list of rules to disable: stop,shift,first,rep")
    # rl_rules (SFT + RL-derived token-bias table; see methods/rl_rules.py)
    ap.add_argument("--rl_table", default="results/rl_delta/olmo2-1b-sft_vs_olmo2-1b-instruct/summary.json")
    ap.add_argument("--rl_k", type=int, default=20)
    ap.add_argument("--rl_beta", type=float, default=1.0)
    ap.add_argument("--rl_scope", choices=["fork", "all"], default="fork")
    ap.add_argument("--rl_fork_thr", type=float, default=None)
    ap.add_argument("--rl_swaps", type=int, default=0, help="use the M most frequent SFT->Instruct swaps as rules")
    ap.add_argument("--rl_bias_json", default=None, help="route 2: bias table from analysis/self_rules.py")
    ap.add_argument("--rl_oracle", action="store_true")
    ap.add_argument("--rl_oracle_model", default="olmo2-1b-instruct")
    # fork_mbr (majority vote over greedy + branches at the F closest-call forks)
    ap.add_argument("--fm_forks", type=int, default=4)
    ap.add_argument("--fm_min_pos", type=int, default=0, help="ignore forks before this response position")
    ap.add_argument("--fm_where", choices=["any", "step_start"], default="any")
    ap.add_argument("--fm_chunk", type=int, default=32, help="branch batch size (use ~8 for 7B models)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    hf_id, fmt = resolve(args.model)
    args.fmt = fmt  # methods that support chat models read this; the others assume "plain"
    task = get_task(args.task)
    rows = read_jsonl(f"data/{args.task}/{args.split}.jsonl")[: args.limit]
    exemplars = (random.Random(args.seed).sample(read_jsonl(f"data/{args.task}/train.jsonl"), args.few_shot)
                 if args.few_shot else [])

    tok = AutoTokenizer.from_pretrained(hf_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    texts = [prompts.build_prompt(fmt, r["question"], exemplars, tok) for r in rows]
    answers = [r["answer"] for r in rows]
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.bfloat16).cuda().eval()

    method = importlib.import_module(f"methods.{args.method}")
    t0 = time.time()
    results = method.run(model, tok, texts, answers, task, args, stop_ids(model.generation_config, tok))
    secs = round(time.time() - t0, 1)

    chosen = [r["paths"][r["chosen"]] for r in results]
    per_problem = [[{**c["score"], "gen_tokens": c["gen_tokens"], "truncated": c["truncated"]}] for c in chosen]
    metrics = aggregate(per_problem)
    metrics.update(method.summarize(results, answers))
    metrics.update({"method": args.method, "model": args.model, "split": args.split, "few_shot": args.few_shot,
                    "args": vars(args), "seconds": secs, "git": git_hash()})

    tag = args.tag or (method.tag(args) if hasattr(method, "tag") else f"k{args.k}")
    out = Path(args.out or f"results/screen/{args.model}/{args.split}/{args.method}_{tag}")
    write_jsonl(out / f"{args.split}.generations.jsonl",
                [{"question": r["question"], "answer": r["answer"], **res} for r, res in zip(rows, results)])
    write_json(out / f"{args.split}.metrics.json", metrics)
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in metrics.items()
                      if k.startswith(("acc", "format", "mean", "trunc")) or k == "seconds"}, indent=1))


if __name__ == "__main__":
    main()
