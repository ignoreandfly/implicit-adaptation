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

## 6. Running the 7B jobs on GCP (added 2026-11-23)

The shared lab server rarely has a nearly free 24 GB GPU, so the 7B experiments are packaged as one script,
`scripts/gcp_7b_jobs.sh`, for a dedicated cloud GPU. It is resumable (finished steps are skipped).

**Machine.** For inference-only jobs (`sample judge fobar forkmbr`) an **L4 (24 GB)** is enough:
`g2-standard-8`, about $0.85/h on demand (spot is roughly a third). For the 7B LoRA training steps (`rft`, the
judge training inside `judge`) an **A100 40 GB** (`a2-highgpu-1g`, about $3.70/h on demand) is safer; 7B training
on an L4 may run out of memory. Use a Deep Learning VM image (CUDA preinstalled) and a >= 200 GB disk.

```bash
# from a machine with gcloud installed and authenticated (gcloud auth login; gcloud config set project <ID>)
# A100 (for the training steps). For an L4 instead use:
#   --machine-type=g2-standard-8 --accelerator=type=nvidia-l4,count=1
# Add --provisioning-model=SPOT for ~1/3 of the price (can be preempted; the script is resumable).
gcloud compute instances create olmo-7b \
  --zone=us-central1-a --machine-type=a2-highgpu-1g \
  --accelerator=type=nvidia-tesla-a100,count=1 \
  --image-family=common-cu124-ubuntu-2204-py310 --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB --maintenance-policy=TERMINATE \
  --metadata=install-nvidia-driver=True
gcloud compute ssh olmo-7b --zone=us-central1-a
```
(Check GPU quota in the chosen region first; image family names change over time —
`gcloud compute images list --project deeplearning-platform-release | grep cu12` lists current ones.)

**On the VM:**
```bash
git clone git@github.com:ignoreandfly/implicit-adaptation.git && cd implicit-adaptation   # needs a deploy key or token
# environments: RUNBOOK section 1 (two envs: hf and vllm)
export HF_HUB_CACHE=$HOME/hf_cache
VPY=~/miniconda3/envs/vllm/bin/python HPY=~/miniconda3/envs/hf/bin/python \
  setsid nohup bash scripts/gcp_7b_jobs.sh > logs/gcp_7b.log 2>&1 < /dev/null &
tail -f logs/gcp_7b.log
```

**What it runs, in order** (`JOBS="sample rft judge fobar forkmbr"`; pick a subset with `JOBS=...`):

| Step | What | Time (L4 / A100, rough) | Compare against |
|---|---|---|---|
| `sample` | 7B SFT: 64 samples on test500 (selection pool) + 8 samples on 1,500 training questions | ~1 h / ~25 min | — |
| `rft` | **Fine-tune 7B SFT on its own correct (gold) and majority-agreeing (label-free) samples**, greedy test500 | ~3 h / ~1 h | 7B SFT greedy 74.6%, 7B Instruct 83.2% |
| `judge` | zero-shot vs trained pairwise self-judge, selection on the test500 pool (k = 4, 8, 16) | ~2 h / ~45 min | majority vote on the same pool |
| `fobar` | backward verification on 7B sample pools (k = 8, 16) | ~40 min / ~20 min | majority vote |
| `forkmbr` | fork-gated MBR on 7B SFT, test500 | ~4 h / ~1.5 h | its own greedy |

Rough total: ~10 GPU-hours on an L4 or ~4 on an A100 (~$10–15 either way on demand; less on spot).

**Results back**: commit the small files (`results/rft_sft7b/`, `results/judge/*.json`, `results/fobar/*7b*`,
`results/screen/olmo2-7b-sft/test500/*/test500.metrics.json`) and push, or `gcloud compute scp` them back.
Do not commit `checkpoints/` (merged 7B models are ~15 GB each) or generations.

**Stop the VM when done** (`gcloud compute instances stop olmo-7b --zone=us-central1-a`; delete it to stop disk
charges too) — an idle A100 still costs ~$3.70/h.

1B reference numbers (same setup) are in README.md: Base + CoT RFT gold 43.7% / label-free 41.6% vs 38.1% (full
test); trained 1B judge +2.2 over majority at k = 8 (only when the top-2 answers are within one vote).
