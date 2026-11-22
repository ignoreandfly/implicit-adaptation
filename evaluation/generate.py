"""Generate on a task split, score with the task verifier, write generations + metrics.

Greedy (pass@1), model by registry name (common/models.py) or HF id:
    python -m evaluation.generate --model olmo2-1b-sft --split test \
        --out results/olmo2-1b-sft/gsm8k/0shot
Few-shot:      add --few_shot 8   (plain: exemplars before the question; chat: earlier turns)
Quick check:   add --limit 50
Fixed subset:  add --subset 500   (500 problems drawn with --subset_seed 0, kept in file order)
Sampling:      add --n_samples 16 --temperature 0.8   (pass@k and maj@k)
vLLM:          add --backend vllm (needs the implicit-vllm env, see README); much faster
               for n_samples > 1. Same prompts, stops, truncation, scoring and output files.
Older GPUs (T4, V100) have no bf16, so --dtype auto falls back to float32.
"""

import argparse
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

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


def stop_ids(gen_config, tok):
    ids = gen_config.eos_token_id
    ids = set(ids if isinstance(ids, list) else [ids]) | {tok.eos_token_id, tok.pad_token_id}
    return {i for i in ids if i is not None}


def subset_indices(n_total, size, seed=0):
    """Fixed random subset of row indices, sorted so file order is kept."""
    if size >= n_total:
        return list(range(n_total))
    return sorted(random.Random(seed).sample(range(n_total), size))


def generate_hf(args, texts, tok, fmt, dtype):
    """HF model.generate. Returns (token ids per prompt as n_samples id lists, seconds)."""
    model = AutoModelForCausalLM.from_pretrained(args.hf_id, torch_dtype=dtype).cuda().eval()
    do_sample = args.temperature > 0
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
    out = []
    for i in range(0, len(texts), bs):
        new = generate(texts[i:i + bs])
        out += [new[j:j + args.n_samples] for j in range(0, len(new), args.n_samples)]
        print(f"{min(i + bs, len(texts))}/{len(texts)}  {time.time() - t0:.0f}s", flush=True)
    return out, time.time() - t0


def generate_vllm(args, texts, tok, fmt, dtype):
    """vLLM offline engine. Same return format as generate_hf.

    Prompts are tokenized here with the HF tokenizer (same add_special_tokens rule as the
    HF path), so both backends see identical input ids. Plain format stops at
    STOP_STRINGS, chat format at EOS only. Raw token ids are returned; the caller decodes,
    truncates and counts them exactly as for HF."""
    from vllm import LLM, SamplingParams  # lazy: the HF path must not need vllm
    from vllm.inputs import TokensPrompt

    # On a shared GPU, vLLM's startup memory profiling fails if another process frees memory
    # meanwhile ("Error in memory profiling"); that is transient, so retry a few times.
    for attempt in range(3):
        try:
            llm = LLM(model=args.hf_id, dtype=str(dtype).replace("torch.", ""), seed=args.seed,
                      gpu_memory_utilization=args.gpu_memory_utilization,
                      max_model_len=args.max_model_len, enforce_eager=args.enforce_eager)
            break
        except RuntimeError as e:
            if attempt == 2:
                raise
            print(f"vLLM startup failed ({e}); retrying in 10s", flush=True)
            time.sleep(10)
    do_sample = args.temperature > 0
    params = SamplingParams(
        n=args.n_samples, temperature=args.temperature if do_sample else 0.0,
        top_p=1.0, top_k=-1, seed=args.seed if do_sample else None,
        max_tokens=args.max_new_tokens,
        stop=list(prompts.STOP_STRINGS) if fmt == "plain" else None)
    enc = tok(texts, add_special_tokens=(fmt == "plain"))["input_ids"]
    t0 = time.time()
    results = llm.generate([TokensPrompt(prompt_token_ids=ids) for ids in enc], params)
    seconds = time.time() - t0
    # Sort each request's completions by sample index (order is not guaranteed).
    return [[list(c.token_ids) for c in sorted(r.outputs, key=lambda c: c.index)]
            for r in results], seconds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="registry name (common/models.py) or HF id")
    ap.add_argument("--format", choices=["plain", "chat"], default=None, help="override the registry")
    ap.add_argument("--task", default="gsm8k")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--subset", type=int, default=None,
                    help="fixed random subset of this many problems (applied before --limit)")
    ap.add_argument("--subset_seed", type=int, default=0, help="seed for --subset; keep 0")
    ap.add_argument("--few_shot", type=int, default=0)
    ap.add_argument("--n_samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=64, help="hf backend only")
    ap.add_argument("--dtype", choices=["auto", *DTYPES], default="auto")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.35,
                    help="vllm only: share of total GPU memory vLLM may use (keep low on shared GPUs)")
    ap.add_argument("--max_model_len", type=int, default=4096, help="vllm only")
    ap.add_argument("--enforce_eager", action="store_true", help="vllm only: skip CUDA graphs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    hf_id, fmt = resolve(args.model, args.format)
    args.hf_id = hf_id
    task = get_task(args.task)
    data_dir = args.data_dir or f"data/{args.task}"
    rows = read_jsonl(f"{data_dir}/{args.split}.jsonl")
    idx = subset_indices(len(rows), args.subset, args.subset_seed) if args.subset else list(range(len(rows)))
    idx = idx[: args.limit]
    rows = [rows[i] for i in idx]

    tok = AutoTokenizer.from_pretrained(hf_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    exemplars = []
    if args.few_shot:
        exemplars = random.Random(args.seed).sample(read_jsonl(f"{data_dir}/train.jsonl"), args.few_shot)
    texts = [prompts.build_prompt(fmt, r["question"], exemplars, tok) for r in rows]
    print(f"{hf_id} [{fmt}] via {args.backend}, {len(rows)} problems; prompt 0:\n{texts[0]}", flush=True)

    if args.n_samples > 1 and not args.temperature > 0:
        raise ValueError("n_samples > 1 needs --temperature > 0")
    dtype = pick_dtype(args.dtype)
    ends = stop_ids(GenerationConfig.from_pretrained(hf_id), tok)

    run = generate_vllm if args.backend == "vllm" else generate_hf
    token_ids, seconds = run(args, texts, tok, fmt, dtype)  # seconds: generation only

    per_problem, gens = [], []
    for i, r, new in zip(idx, rows, token_ids):
        outs = [prompts.truncate(d, fmt) for d in tok.batch_decode(new, skip_special_tokens=True)]
        # Generated length = tokens before the first EOS/pad (stop strings also end in padding).
        lens = [next((t for t, x in enumerate(ids) if x in ends), len(ids)) for ids in new]
        scores = [{**task.verify(o, r["answer"]), "gen_tokens": n,
                   "truncated": n >= args.max_new_tokens} for o, n in zip(outs, lens)]
        per_problem.append(scores)
        gens.append({"idx": i, "question": r["question"], "answer": r["answer"],
                     "num_steps": r["num_steps"], "outputs": outs, "scores": scores})

    metrics = aggregate(per_problem)
    metrics.update({"model": args.model, "hf_id": hf_id, "format": fmt, "task": args.task,
                    "split": args.split, "subset": args.subset, "few_shot": args.few_shot,
                    "temperature": args.temperature, "n_samples": args.n_samples,
                    "max_new_tokens": args.max_new_tokens, "seed": args.seed,
                    "dtype": str(dtype).replace("torch.", ""), "limit": args.limit,
                    "backend": args.backend, "gpu": torch.cuda.get_device_name(),
                    "seconds": round(seconds, 1), "git": git_hash()})
    if args.backend == "vllm":
        import vllm
        metrics.update({"vllm_version": vllm.__version__,
                        "gpu_memory_utilization": args.gpu_memory_utilization})
    out = Path(args.out)
    write_jsonl(out / f"{args.split}.generations.jsonl", gens)
    write_json(out / f"{args.split}.metrics.json", metrics)
    print({k: round(v, 3) for k, v in metrics.items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
