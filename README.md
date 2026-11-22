# Implicit Adaptation below 3B Parameters

COMS 4705 final project. Hewitt et al. ("Instruction Following without Instruction Tuning")
showed that a small hand-written program that edits the next-token distribution, with no
training, can make a base model follow instructions. We ask whether that kind of
rule-based adapter, combined with inference-time scaling, can bring out step-by-step
reasoning in models of 1B parameters or fewer, and how close it gets to models that went
through SFT and RL.

We evaluate on GSM8K first. Task list and open decisions: [ROADMAP.md](ROADMAP.md).

## Models

We use OLMo 2 1B because it releases every post-training stage, so we can see what SFT,
DPO and RL each add, and its pretraining data is open.

| Name (`--model`) | Checkpoint | Post-training | Prompt |
|---|---|---|---|
| `olmo2-1b-base` | `allenai/OLMo-2-0425-1B` | none | plain |
| `olmo2-1b-sft` | `allenai/OLMo-2-0425-1B-SFT` | SFT (Tülu 3 mix, includes math CoT) | chat |
| `olmo2-1b-dpo` | `allenai/OLMo-2-0425-1B-DPO` | SFT + DPO | chat |
| `olmo2-1b-instruct` | `allenai/OLMo-2-0425-1B-Instruct` | SFT + DPO + RLVR (math) | chat |

The names are defined in `common/models.py`.

## Layout

```
common/      model registry, prompt construction (plain / chat), jsonl I/O
tasks/       gsm8k.py: loader (writes data/gsm8k/*.jsonl) + answer verifier
evaluation/  generate.py (HF or vLLM generation + scoring), metrics.py (accuracy, pass@k, maj@k, format, steps, length)
notebooks/   colab_baselines.ipynb: all baseline runs on Colab, results saved to Drive
adapters/    rule-based logits processors (TODO: Hewitt adapter)
analysis/    summarize.py -> results/summary.csv
data/        downloaded splits (gitignored)
results/     generations + metrics per run
```

## Running on Colab

`notebooks/colab_baselines.ipynb` runs a 50-question smoke test per model, then the full test
set. Finished runs are skipped, so re-running after a disconnect resumes.

- **VS Code + Google Colab extension:** open the notebook, *Select Kernel > Colab > New Colab
  Server*, pick a GPU. The kernel runs on the Colab machine, so the notebook clones the repo
  there. Put tokens in `.env` (copy `.env.example`; gitignored) and right-click it > *Upload
  to Colab*; without it the notebook prompts for the GitHub token.
  For results on Drive, run *Colab: Mount Google Drive to Server...* first; otherwise they land
  in `/content/results` and are zipped at the end for download from the *Contents* view.
- **Colab web UI:** pick a GPU runtime, add a `GITHUB_TOKEN` secret, run all cells. Results
  are saved to `MyDrive/implicit-adaptation/`.

## Quick start (any GPU machine)

```bash
PY=~/miniconda3/envs/ICCV/bin/python    # temporary env: torch 2.7, transformers 4.57
$PY -m pip install -r requirements.txt

$PY -m tasks.gsm8k --out data/gsm8k     # train (7,273) / dev (200) / test (1,319)
$PY -m evaluation.generate --model olmo2-1b-base --few_shot 8 \
    --out results/olmo2-1b-base/gsm8k/8shot
$PY -m evaluation.generate --model olmo2-1b-sft --out results/olmo2-1b-sft/gsm8k/0shot
$PY -m analysis.summarize results       # one table over all runs
```

Add `--limit 50` to `evaluation.generate` for a quick check. `--dtype auto` uses bf16 on
L4/A100 and float32 on older GPUs (T4, V100).

## Sampling with vLLM (`--backend vllm`)

`evaluation.generate --backend hf` (default) uses HF `model.generate`, which drops to batch size 1 at
`--n_samples 64`. `--backend vllm` runs the same prompts (incl. chat template and few-shot), the same
stops (plain: `STOP_STRINGS`; chat: EOS only), truncation, scoring and output files, with
`SamplingParams(n=n_samples, temperature, top_p=1.0, seed, max_tokens=max_new_tokens)`. Prompts are
tokenized with the HF tokenizer, and the returned token ids are decoded and counted like the HF path, so
`gen_tokens` and `truncated` mean the same thing. vLLM is imported only on that path, so the HF path still
runs in an env without it.

vLLM needs its own env (pins its own torch). On the lab server (driver 535, CUDA 12.2) we use
vLLM 0.9.2 = torch 2.7.0+cu126, which runs through CUDA minor-version compatibility:

```bash
~/miniconda3/bin/conda create -y -n implicit-vllm python=3.11
VPY=~/miniconda3/envs/implicit-vllm/bin/python
$VPY -m pip install --no-cache-dir vllm==0.9.2 "transformers==4.53.3" "huggingface-hub<1.0" \
    "accelerate<1.8" "datasets<3.7" pandas tabulate   # ~8.4 GB
```

(vLLM 0.9.2 breaks with transformers >= 4.54, and in vLLM <= 0.8.x `gpu_memory_utilization` counts
other processes' memory too, so it fails on a shared GPU.)

```bash
# 500-problem test subset, 64 samples per problem; GPU 1 only, vLLM may take 35% of its memory
CUDA_VISIBLE_DEVICES=1 $VPY -m evaluation.generate --backend vllm --model olmo2-1b-sft \
    --subset 500 --n_samples 64 --temperature 0.8 --gpu_memory_utilization 0.35 \
    --out results/olmo2-1b-sft/gsm8k/0shot_sub500_n64_t0.8
GPU=1 bash scripts/gsm8k_sampling.sh    # all 5 configs (base 0/8-shot, SFT, DPO, Instruct)
```

- `--subset N` picks a fixed random subset of the split: `random.Random(0).sample(range(len(split)), N)`,
  sorted, so the same 500 test problems for every model (`--subset_seed` changes the seed; keep 0).
  It is applied before `--limit`. Each generations row stores its `idx` in the full split.
- `--gpu_memory_utilization` (default 0.35) is the share of the GPU's total memory this vLLM instance may
  use (weights + KV cache). Check `nvidia-smi` first; the 1B models need ~3 GB of it for weights.
- vLLM flags: `--max_model_len` (default 4096), `--enforce_eager` (no CUDA graphs). `--batch_size` is HF only.
- "Error in memory profiling" at startup means another process freed GPU memory while vLLM was
  profiling; `generate.py` retries startup up to 3 times.
- Greedy vLLM matches greedy HF up to bf16 noise: on the first 100 test problems it agrees with HF as often
  as HF at batch 64 agrees with HF at batch 1 (see ROADMAP).

## Conventions

- Base models: `Question: {q}\nAnswer:`, generation cut at a blank line. Post-trained models:
  their chat template, with the question plus "Solve the problem step by step. End your
  response with "The answer is N."", stopping at EOS only. Reference solutions are the GSM8K
  rationale on one line, without `<<calc>>` notes, ending in `Therefore, the answer is N.`
- Two accuracies: **strict** (`the answer is N`) and **lenient** (first `answer ...`,
  `#### N` or `\boxed{N}`, else the last number). The gap shows how much a model loses to format alone.
- Numbers are normalized before comparing (`$`, commas, trailing `.0`).
- Raw generations are saved, so a parser fix means re-scoring, not re-generating.
