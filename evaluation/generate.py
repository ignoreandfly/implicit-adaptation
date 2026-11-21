"""Generate on a task split, score with the task verifier, write generations + metrics.

Greedy (pass@1), model by registry name (common/models.py) or HF id:
    python -m evaluation.generate --model olmo2-1b-sft --split test \
        --out results/olmo2-1b-sft/gsm8k/0shot
Few-shot:      add --few_shot 8   (plain: exemplars before the question; chat: earlier turns)
Quick check:   add --limit 50
Sampling:      add --n_samples 16 --temperature 0.8   (pass@k and maj@k)
Older GPUs (T4, V100) have no bf16, so --dtype auto falls back to float32.
"""

import argparse
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import prompts
from common.io import git_hash, read_jsonl, write_json, write_jsonl
from common.models import resolve
from evaluation.metrics import aggregate
from tasks import get_task

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def pick_dtype(name):
    if name != "auto":
        return DTYPES[name]
    # bf16 needs compute capability >= 8 (A100, L4, ...); is_bf16_supported() also
    # reports True for emulated bf16 on T4, which is slow, so check the hardware.
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float32


def stop_ids(model, tok):
    ids = model.generation_config.eos_token_id
    ids = set(ids if isinstance(ids, list) else [ids]) | {tok.eos_token_id, tok.pad_token_id}
    return {i for i in ids if i is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="registry name (common/models.py) or HF id")
    ap.add_argument("--format", choices=["plain", "chat"], default=None, help="override the registry")
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--few_shot", type=int, default=0)
    ap.add_argument("--n_samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--dtype", choices=["auto", *DTYPES], default="auto")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    hf_id, fmt = resolve(args.model, args.format)
    task = get_task(args.task)
    data_dir = args.data_dir or f"data/{args.task}"
    rows = read_jsonl(f"{data_dir}/{args.split}.jsonl")[: args.limit]

    tok = AutoTokenizer.from_pretrained(hf_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    exemplars = []
    if args.few_shot:
        exemplars = random.Random(args.seed).sample(read_jsonl(f"{data_dir}/train.jsonl"), args.few_shot)
    texts = [prompts.build_prompt(fmt, r["question"], exemplars, tok) for r in rows]
    print(f"{hf_id} [{fmt}], {len(rows)} problems; prompt 0:\n{texts[0]}", flush=True)

    dtype = pick_dtype(args.dtype)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=dtype).cuda().eval()
    ends = stop_ids(model, tok)

    do_sample = args.temperature > 0
    if args.n_samples > 1 and not do_sample:
        raise ValueError("n_samples > 1 needs --temperature > 0")
    bs = max(1, args.batch_size // args.n_samples) if args.n_samples > 1 else args.batch_size
    stop_kw = {"stop_strings": prompts.STOP_STRINGS, "tokenizer": tok} if fmt == "plain" else {}

    def generate(batch):
        """Token ids generated for each prompt in batch (n_samples rows per prompt).
        On CUDA out-of-memory, splits the batch in half and retries."""
        # Chat templates already contain any special tokens, so don't add a second BOS.
        enc = tok(batch, return_tensors="pt", padding=True,
                  add_special_tokens=(fmt == "plain")).to("cuda")
        try:
            with torch.no_grad():
                gen = model.generate(
                    **enc, max_new_tokens=args.max_new_tokens, do_sample=do_sample,
                    temperature=args.temperature if do_sample else None,
                    top_p=1.0 if do_sample else None, top_k=0 if do_sample else None,
                    num_return_sequences=args.n_samples, pad_token_id=tok.pad_token_id, **stop_kw)
        except torch.cuda.OutOfMemoryError:
            if len(batch) == 1:
                raise
            del enc
            torch.cuda.empty_cache()
            h = len(batch) // 2
            print(f"OOM at batch {len(batch)}; retrying as {h} + {len(batch) - h}", flush=True)
            return generate(batch[:h]) + generate(batch[h:])
        return gen[:, enc.input_ids.shape[1]:].tolist()

    t0 = time.time()
    outputs, lengths = [], []
    for i in range(0, len(texts), bs):
        new = generate(texts[i:i + bs])
        dec = tok.batch_decode(new, skip_special_tokens=True)
        # Generated length = tokens before the first EOS/pad (stop strings also end in padding).
        lens = [next((t for t, x in enumerate(ids) if x in ends), len(ids)) for ids in new]
        for j in range(0, len(dec), args.n_samples):
            outputs.append([prompts.truncate(d, fmt) for d in dec[j:j + args.n_samples]])
            lengths.append(lens[j:j + args.n_samples])
        print(f"{min(i + bs, len(texts))}/{len(texts)}  {time.time() - t0:.0f}s", flush=True)

    per_problem, gens = [], []
    for r, outs, lens in zip(rows, outputs, lengths):
        scores = [{**task.verify(o, r["answer"]), "gen_tokens": n,
                   "truncated": n >= args.max_new_tokens} for o, n in zip(outs, lens)]
        per_problem.append(scores)
        gens.append({"question": r["question"], "answer": r["answer"], "num_steps": r["num_steps"],
                     "outputs": outs, "scores": scores})

    metrics = aggregate(per_problem)
    metrics.update({"model": args.model, "hf_id": hf_id, "format": fmt, "task": args.task,
                    "split": args.split, "few_shot": args.few_shot, "temperature": args.temperature,
                    "n_samples": args.n_samples, "max_new_tokens": args.max_new_tokens,
                    "dtype": str(dtype).replace("torch.", ""), "limit": args.limit,
                    "gpu": torch.cuda.get_device_name(), "seconds": round(time.time() - t0, 1),
                    "git": git_hash()})
    out = Path(args.out)
    write_jsonl(out / f"{args.split}.generations.jsonl", gens)
    write_json(out / f"{args.split}.metrics.json", metrics)
    print({k: round(v, 3) for k, v in metrics.items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
