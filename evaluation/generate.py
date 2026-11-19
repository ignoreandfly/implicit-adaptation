"""Generate on a task split, score with the task verifier, write generations + metrics.

Greedy (pass@1):
    python -m evaluation.generate --model Qwen/Qwen2.5-0.5B --task arithmetic --split test_id \
        --out results/qwen0.5b/base_zero_shot
Few-shot base model:   add --few_shot 3
Trained LoRA adapter:  add --adapter runs/<run>/adapter
pass@k:                add --n_samples 64 --temperature 0.8
"""

import argparse
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import prompts
from common.io import git_hash, read_jsonl, write_json, write_jsonl
from evaluation.metrics import aggregate
from tasks import get_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--task", default="arithmetic")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--split", default="test_id")
    ap.add_argument("--few_shot", type=int, default=0)
    ap.add_argument("--n_samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    task = get_task(args.task)
    data_dir = args.data_dir or f"data/{args.task}"
    rows = read_jsonl(f"{data_dir}/{args.split}.jsonl")[: args.limit]

    prefix = ""
    if args.few_shot:
        ex = random.Random(args.seed).sample(read_jsonl(f"{data_dir}/train.jsonl"), args.few_shot)
        prefix = prompts.few_shot_prefix(ex)
    texts = [prefix + prompts.qa_prompt(r["question"]) for r in rows]

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).cuda().eval()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()

    do_sample = args.temperature > 0
    if args.n_samples > 1 and not do_sample:
        raise ValueError("n_samples > 1 needs --temperature > 0")
    bs = max(1, args.batch_size // args.n_samples) if args.n_samples > 1 else args.batch_size

    t0 = time.time()
    outputs = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=args.max_new_tokens, do_sample=do_sample,
                temperature=args.temperature if do_sample else None, top_p=1.0 if do_sample else None,
                top_k=0 if do_sample else None,
                num_return_sequences=args.n_samples, pad_token_id=tok.pad_token_id,
                stop_strings=prompts.STOP_STRINGS, tokenizer=tok)
        dec = tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
        for j in range(0, len(dec), args.n_samples):
            outputs.append([prompts.truncate(d) for d in dec[j:j + args.n_samples]])
        print(f"{min(i + bs, len(texts))}/{len(texts)}  {time.time() - t0:.0f}s", flush=True)

    per_problem, gens = [], []
    for r, outs in zip(rows, outputs):
        scores = [task.verify(o, r["answer"]) for o in outs]
        per_problem.append(scores)
        gens.append({"question": r["question"], "answer": r["answer"], "num_steps": r["num_steps"],
                     "outputs": outs, "scores": scores})

    metrics = aggregate(per_problem)
    metrics.update({"model": args.model, "adapter": args.adapter, "task": args.task, "split": args.split,
                    "few_shot": args.few_shot, "temperature": args.temperature,
                    "seconds": round(time.time() - t0, 1), "git": git_hash()})
    out = Path(args.out)
    write_jsonl(out / f"{args.split}.generations.jsonl", gens)
    write_json(out / f"{args.split}.metrics.json", metrics)
    print({k: v for k, v in metrics.items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
