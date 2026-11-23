"""Inference-only test of SSFT global forking tokens (Jia et al., ICLR 2026, arXiv 2510.05132) on GSM8K.

Uses the released checkpoints, no training:
  shengjia-toronto/ssft-32B-N6                  SSFT model with forking tokens <think1> ... <think6>
  shengjia-toronto/sft_mixed_32b_thinktags_GAS4 the paper's matched baseline: plain SFT on the same mixed traces
Prompt format follows the authors' LightEval code (SSFT repo, lighteval/src/lighteval/models/vllm/vllm_model.py):
chat template with the generation prompt, then the suffix  "<|im_start|><think{i}>\n"  for forking token i.

Modes (one jsonl row per question with every path; metrics computed by analysis/equal_compute.py):
  tags       one path per forking token i = 1..N         (Cons@N: the paper's parallel mode)
  repeat     n paths all with the SAME suffix (--tag i)   (temperature-only diversity: the control)
  plain      n paths with no forking-token suffix         (for the baseline model)
--temperature 0 gives greedy paths (in "tags" mode: deterministic diversity from the tokens alone).

    python -m analysis.ssft_infer --model shengjia-toronto/ssft-32B-N6 --mode tags --n_tags 6 \
        --temperature 0.7 --split test500 --tp 2 --out results/ssft/ssft32b_tags6_T0.7
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from common.io import read_jsonl, write_json, write_jsonl
from tasks import get_task

INSTR = "Please reason step by step, and put your final answer within \\boxed{}."
BOXED = re.compile(r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")


def final_text(t):
    """Text after the reasoning block if present; the boxed answer (last one) if present."""
    if "</think>" in t:
        t = t.split("</think>")[-1]
    b = BOXED.findall(t)
    return f"The answer is {b[-1]}" if b else t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["tags", "repeat", "plain"], required=True)
    ap.add_argument("--n_tags", type=int, default=6)
    ap.add_argument("--tag", type=int, default=1)
    ap.add_argument("--n", type=int, default=6, help="paths per question for repeat / plain")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_tokens", type=int, default=8192)
    ap.add_argument("--split", default="test500")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    task = get_task("gsm8k")
    rows = read_jsonl(f"data/gsm8k/{args.split}.jsonl")[: args.limit]
    llm = LLM(model=args.model, dtype="bfloat16", tensor_parallel_size=args.tp, max_model_len=args.max_tokens + 1024,
              gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    base = [tok.apply_chat_template([{"role": "user", "content": f"{r['question']}\n{INSTR}"}], tokenize=False,
                                    add_generation_prompt=True) for r in rows]
    if args.mode == "tags":
        suffixes = [f"<|im_start|><think{i}>\n" for i in range(1, args.n_tags + 1)]
        reqs = [(qi, s, b + s, 1) for qi, b in enumerate(base) for s in suffixes]
    elif args.mode == "repeat":
        s = f"<|im_start|><think{args.tag}>\n"
        reqs = [(qi, s, b + s, args.n) for qi, b in enumerate(base)]
    else:
        reqs = [(qi, "", b, args.n) for qi, b in enumerate(base)]
    groups = {}
    for qi, s, prompt, n in reqs:
        groups.setdefault(n, []).append((qi, s, prompt))
    paths = {qi: [] for qi in range(len(rows))}
    for n, items in groups.items():
        sp = SamplingParams(n=n, temperature=args.temperature, top_p=args.top_p if args.temperature > 0 else 1.0,
                            max_tokens=args.max_tokens, seed=args.seed)
        outs = llm.generate([p for _, _, p in items], sp)
        for (qi, s, _), o in zip(items, outs):
            for c in o.outputs:
                sc = task.verify(final_text(c.text), rows[qi]["answer"])
                paths[qi].append({"suffix": s, "score": sc, "gen_tokens": len(c.token_ids),
                                  "truncated": c.finish_reason == "length", "output": c.text})
    res = []
    for qi, r in enumerate(rows):
        ps = paths[qi]
        v = Counter(p["score"]["lenient_answer"] for p in ps if p["score"]["lenient_answer"] is not None)
        maj = v.most_common(1)[0][0] if v else None
        chosen = next((j for j, p in enumerate(ps) if p["score"]["lenient_answer"] == maj), 0)
        res.append({"question": r["question"], "answer": r["answer"], "paths": ps, "chosen": chosen})
    out = Path(args.out)
    write_jsonl(out / f"{args.split}.generations.jsonl", res)
    n = len(res)
    m = {"acc_majority": sum(r["paths"][r["chosen"]]["score"]["lenient_correct"] for r in res) / n,
         "acc_single_path_mean": sum(p["score"]["lenient_correct"] for r in res for p in r["paths"]) /
         sum(len(r["paths"]) for r in res),
         "acc_oracle": sum(any(p["score"]["lenient_correct"] for p in r["paths"]) for r in res) / n,
         "truncation_rate": sum(p["truncated"] for r in res for p in r["paths"]) / sum(len(r["paths"]) for r in res),
         "mean_gen_tokens": sum(p["gen_tokens"] for r in res for p in r["paths"]) / sum(len(r["paths"]) for r in res),
         "args": vars(args)}
    if args.mode == "tags":
        m["acc_per_tag"] = {f"think{i + 1}": sum(r["paths"][i]["score"]["lenient_correct"] for r in res) / n
                            for i in range(args.n_tags)}
    write_json(out / f"{args.split}.metrics.json", m)
    print(json.dumps({k: v for k, v in m.items() if k != "args"}, indent=2))


if __name__ == "__main__":
    main()
