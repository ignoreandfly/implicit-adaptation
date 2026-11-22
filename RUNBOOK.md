# Runbook: picking up experiments on another GPU / machine

For anyone (a teammate, or a Claude session) continuing this project on a different GPU. It covers setup, the
pending experiments with exact commands, memory needs, and the pitfalls we hit. Project context and all results
are in [README.md](README.md); ideas to revisit at larger scale are in [SCALE_NOTES.md](SCALE_NOTES.md).

Last updated: 2026-11-23.

## 1. Setup

```bash
git clone git@github.com:ignoreandfly/implicit-adaptation.git && cd implicit-adaptation
```

Two Python environments are used, because vLLM pins an older transformers:

| Env | Used for | Versions that work |
|---|---|---|
| `hf` (HF + training) | `methods.run`, `train.*`, `analysis.*` | torch 2.7.1, transformers 4.57.1, peft 0.17.0 (+ `requirements.txt`) |
| `vllm` (fast sampling) | `evaluation.generate --backend vllm`, `train.fork_pairs`, `analysis.fobar` | vllm 0.9.2, torch 2.7.0+cu126, transformers 4.53.3 (vllm 0.9.2 breaks with transformers >= 4.54) |

```bash
conda create -y -n hf python=3.10 && conda activate hf
pip install torch==2.7.1 transformers==4.57.1 peft==0.17.0 -r requirements.txt
conda create -y -n vllm python=3.11 && conda activate vllm
pip install vllm==0.9.2 transformers==4.53.3 pandas numpy
```

Data (GSM8K, ~10 MB; never commit it) and the fixed 500-problem test subset:

```bash
python -m tasks.gsm8k --out data/gsm8k        # train 7,273 / dev 200 / test 1,319
python -c "
from evaluation.generate import subset_indices
from common.io import read_jsonl, write_jsonl
rows = read_jsonl('data/gsm8k/test.jsonl'); idx = subset_indices(len(rows), 500, 0)
write_jsonl('data/gsm8k/test500.jsonl', [{**rows[i], 'idx': i} for i in idx])"
```

Models are downloaded from Hugging Face on first use (registry: `common/models.py`). 1B checkpoints are ~3–6 GB
each; 7B SFT/DPO are 14.6 GB (`.bin`), 7B Instruct 14.6 GB, 7B Base 29 GB (fp32). Put them on a big disk:
`export HF_HUB_CACHE=/path/with/space`.

## 2. Memory needs (24 GB GPU, e.g. RTX 3090 / L4)

| Job | GPU memory |
|---|---|
| 1B greedy eval (`methods.run`, `--batch_problems 1`) | ~4–6 GB |
| 1B two-model decoding (fork-only / oracle, `rl_rules --rl_oracle`) | ~8–13 GB |
| 1B LoRA training (`train.fork_lora`, `train.rft_lora`) | ~6–8 GB |
| 1B vLLM sampling (`--gpu_memory_utilization 0.3`) | ~7 GB |
| 7B vLLM sampling / FOBAR (`--gpu_memory_utilization 0.8`) | ~20 GB (needs a nearly free GPU) |
| 7B HF decoding (fork-MBR, `--fm_chunk 8 --batch_problems 1`) | ~18–20 GB |

## 3. Pending experiments (in priority order)

Each command writes metrics to `results/...` (small JSON, committed) and generations to `*.generations.jsonl`
(gitignored). Add `CUDA_VISIBLE_DEVICES=<gpu>`.

**A. 7B FOBAR (backward verification) on 7B SFT sample pools** — needs the 7B SFT n=64 generations
(`results/olmo2-7b-sft/gsm8k/0shot_sub500_n64_t0.7/test.generations.jsonl`, gitignored: regenerate with B0 below
if missing).
```bash
# B0 (only if the generations file is missing; ~30-40 min):
HF_HUB_CACHE=... python -m evaluation.generate --backend vllm --model olmo2-7b-sft --subset 500 --n_samples 64 \
  --temperature 0.7 --gpu_memory_utilization 0.8 --out results/olmo2-7b-sft/gsm8k/0shot_sub500_n64_t0.7
for k in 8 16; do
  HF_HUB_CACHE=... python -m analysis.fobar --model olmo2-7b-sft --pool sampling --k $k --gpu_memory_utilization 0.8 \
    --sampling_gens results/olmo2-7b-sft/gsm8k/0shot_sub500_n64_t0.7/test.generations.jsonl \
    --out results/fobar/olmo2-7b-sft_sampling_k$k.json
done
```
Compare `fobar` vs `majority` in the output (1B: no gain on sample pools).

**B. Fork-gated MBR on 7B SFT (test500)** — ~3–5 h on a free GPU.
```bash
HF_HUB_CACHE=... python -m methods.run --method fork_mbr --model olmo2-7b-sft --split test500 --max_new_tokens 512 \
  --fm_forks 8 --batch_problems 1 --fm_chunk 8
```
Reports `acc_F1..F8`, rescue/harm and cost vs its own greedy (1B: +4.2 at ~5x).

**C. Stage-ladder token analysis at 7B (SFT → Instruct)** — `analysis.rl_delta` loads both models at once
(2 × 15 GB), so on a 24 GB GPU it needs a two-pass rewrite (score with SFT, save top-k log-probs, then Instruct).
Not implemented yet; 1B version: `scripts/rl_delta_step1.sh`.

**D. A second dataset (MATH500 or SVAMP)** for the headline results (sampling curves per stage, adaptive voting).
Needs a new `tasks/<name>.py` with the same interface as `tasks/gsm8k.py` (loader + `verify` + `normalize_number`),
registered in `tasks/__init__.py`; then `scripts/gsm8k_sampling.sh` with `DATA=data/<name>`.

**E. Fork-LoRA follow-ups (1B)** — the method and its controls:
```bash
# data (vLLM): fork pairs on N training questions
python -m train.fork_pairs --model olmo2-1b-sft --n 1000 --out data/fork_pairs/olmo2-1b-sft_train1000.jsonl
#   for Base + zero-shot CoT add:  --model olmo2-1b-base --prefix " Let's think step by step." --max_tokens 320
# train (HF): bounded token-level DPO at forks; --label majority|gold, --shuffle for the control
python -m train.fork_lora --pairs <pairs.jsonl> --model olmo2-1b-sft --loss dpo --lr 1e-4 --beta 0.5 --epochs 2 \
  --label gold --out <dir>
# evaluate (merged model): SFT -> methods.run --method rl_rules --rl_beta 0 --model <dir> --format chat ...
#                          Base -> methods.run --method greedy --model <dir> --prefix " Let's think step by step." ...
# fork-only inference: methods.run --method rl_rules --model <base model> --rl_oracle --rl_oracle_model <dir> ...
# RFT baseline (fine-tune on own correct samples): train/rft_lora.py, see scripts/queue_rft_baseline.sh
```
Lessons so far: unbounded `--loss adv` at lr 1e-4 destroys the model; lr <= 2e-5 changes nothing (bf16); use
`--loss dpo --lr 1e-4`. Always run the `--shuffle` control and an `--lr 0` sanity check.

## 4. Pitfalls we hit (please read)

- **Shared GPUs + vLLM start-up**: vLLM 0.9.2 fails with "Engine core initialization failed" when other processes'
  memory changes during its start-up profiling. Retry with waits (see `scripts/queue_base_fork_lora.sh`) or use the
  HF path. Check free memory first: `nvidia-smi --query-gpu=index,memory.free --format=csv`.
- **Waiting on other jobs**: never wait with `pgrep -f "<pattern>"` — the pattern also matches the shell that
  launched the waiter, so it waits forever (this happened four times). Wait on a PID, a result file, or free GPU
  memory (`pick_gpu` in `scripts/queue_gpu_wait.sh`). Likewise never `pkill -f` a pattern that appears in your own
  command line.
- **Detach long jobs** with `setsid nohup bash script.sh > log 2>&1 < /dev/null &` so they survive the terminal or
  session ending (a 7B run died that way).
- **Greedy decoding at 1B is numerically fragile**: batch size / code path alone moves SFT greedy accuracy between
  56.6% and 60.4% on test500. Only compare runs from the same pipeline, and use paired tests (McNemar).
- **Dev vs test**: tune on `dev` (200), report on `test500` or the full `test` (1,319). Several dev gains (e.g.
  self-certainty voting +3) vanished on test.
- **Verifier**: `tasks/gsm8k.py` treats absurd numbers (inf / NaN) as "no answer" (a 400-digit output once crashed a
  run).

## 5. Reporting back

Commit code and small metrics files (`results/**/*.metrics.json`, `summary.json`, `fork_lora_config.json`), never
`data/`, generations, model weights or PDFs; end commit messages with the co-author line used in this repo. Add new
numbers to the relevant README section, and failed-at-1B ideas to `SCALE_NOTES.md`.
