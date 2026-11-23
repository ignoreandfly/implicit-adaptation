"""Training-free "global forks" (SSFT idea without training): force an opening after the chat prompt, then sample.

Stage `pool`: force every opening in POOL on N training questions and sample n paths each (vLLM). Scores and the
extracted answers are saved per (question, opening, path); the opening set is chosen from them afterwards, with gold
or label-free targets. The empty opening "" is plain temperature sampling, in the same pipeline.

    CUDA_VISIBLE_DEVICES=6 $VPY -m analysis.global_forks --n_questions 500 --n 6 --out results/global_forks/pool
"""

import argparse
import json
import random
import time
from pathlib import Path

from common import prompts
from common.io import read_jsonl, write_json, write_jsonl
from common.models import resolve
from tasks import get_task

NATURAL = ["First, calculate", "First, let's", "To find", "First, determine", "Let's denote", "First, we",
           "To calculate", "Let's break", "Step 1:", "Let's start", "Since", "The"]
STRATEGY = ["Let x be", "Let's define the variables and write an equation.", "Let's work backwards from what is asked.",
            "Here is what we know:", "The question asks for", "Let's solve this step by step, checking each calculation.",
            "We can write this as a short Python program:\n```python\n", "Let's make a table of the quantities:",
            "Let's track the units in each step.", "First, estimate the answer.",
            "Let's write one equation per sentence of the problem.", "Breaking the problem into smaller questions:"]
POOL = [""] + NATURAL + STRATEGY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="olmo2-1b-sft")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n_questions", type=int, default=500)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    task = get_task("gsm8k")
    rows = read_jsonl(f"data/gsm8k/{args.split}.jsonl")
    idx = sorted(random.Random(args.seed).sample(range(len(rows)), min(args.n_questions, len(rows))))
    hf_id, fmt = resolve(args.model)
    assert fmt == "chat"
    llm = LLM(model=hf_id, dtype="bfloat16", gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=4096,
              seed=args.seed, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    keys, reqs = [], []
    for i in idx:
        base = prompts.build_prompt(fmt, rows[i]["question"], [], tok)
        for k, o in enumerate(POOL):
            keys.append((i, k))
            reqs.append({"prompt_token_ids": tok(base + o, add_special_tokens=False).input_ids})
    sp = SamplingParams(n=args.n, temperature=args.temperature, top_p=1.0, max_tokens=args.max_tokens, seed=args.seed)
    t0 = time.time()
    outs = llm.generate(reqs, sp)
    secs = time.time() - t0
    recs = []
    for (i, k), o in zip(keys, outs):
        paths = []
        for c in o.outputs:
            text = prompts.truncate(POOL[k] + c.text, fmt)
            s = task.verify(text, rows[i]["answer"])
            paths.append({"answer": s["lenient_answer"], "correct": bool(s["lenient_correct"]), "tokens": len(c.token_ids),
                          "truncated": c.finish_reason == "length", "output": text})
        recs.append({"idx": i, "gold": rows[i]["answer"], "opening": k, "paths": paths})
    out = Path(args.out)
    write_jsonl(out / f"{args.split}.generations.jsonl", recs)
    summ = {}
    for k, o in enumerate(POOL):
        ps = [p for r in recs if r["opening"] == k for p in r["paths"]]
        summ[o or "(none)"] = {"acc_single": sum(p["correct"] for p in ps) / len(ps),
                               "tokens": sum(p["tokens"] for p in ps) / len(ps),
                               "truncated": sum(p["truncated"] for p in ps) / len(ps)}
    write_json(out / f"{args.split}.metrics.json", {"openings": summ, "seconds": secs, "args": vars(args), "idx": idx})
    for o, v in sorted(summ.items(), key=lambda x: -x[1]["acc_single"]):
        print(f"{v['acc_single']:.3f}  tok {v['tokens']:.0f}  trunc {v['truncated']:.3f}  {o!r}")
    print("seconds", round(secs))


if __name__ == "__main__":
    main()
