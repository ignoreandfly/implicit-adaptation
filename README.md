# Implicit Adaptation below 3B Parameters

COMS 4705 final project. Hewitt et al. ("Instruction Following without Instruction Tuning")
showed that a small hand-written program that edits the next-token distribution, with no
training, can make a base model follow instructions. We ask whether that kind of
rule-based adapter, combined with inference-time scaling, can bring out step-by-step
reasoning in models of 1B parameters or fewer, and how close it gets to models that went
through SFT and RL.

We evaluate on GSM8K first. Task list and open decisions: [ROADMAP.md](ROADMAP.md).

## Models

OLMo 2 1B is the main family because it releases every post-training stage, so we can
see what SFT, DPO and RL each add. Qwen2.5-0.5B is the second family.

| Name (`--model`) | Checkpoint | Post-training | Prompt |
|---|---|---|---|
| `olmo2-1b-base` | `allenai/OLMo-2-0425-1B` | none | plain |
| `olmo2-1b-sft` | `allenai/OLMo-2-0425-1B-SFT` | SFT (Tülu 3 mix, includes math CoT) | chat |
| `olmo2-1b-dpo` | `allenai/OLMo-2-0425-1B-DPO` | SFT + DPO | chat |
| `olmo2-1b-instruct` | `allenai/OLMo-2-0425-1B-Instruct` | SFT + DPO + RLVR (math) | chat |
| `qwen2.5-0.5b-base` | `Qwen/Qwen2.5-0.5B` | none | plain |
| `qwen2.5-0.5b-instruct` | `Qwen/Qwen2.5-0.5B-Instruct` | SFT + DPO + GRPO (learned reward model) | chat |

The names are defined in `common/models.py`.

## Layout

```
common/      model registry, prompt construction (plain / chat), jsonl I/O
tasks/       gsm8k.py: loader (writes data/gsm8k/*.jsonl) + answer verifier
evaluation/  generate.py (HF generate + scoring), metrics.py (accuracy, pass@k, maj@k, format, steps, length)
notebooks/   colab_baselines.ipynb: all baseline runs on Colab, results saved to Drive
adapters/    rule-based logits processors (TODO: Hewitt adapter)
analysis/    summarize.py -> results/summary.csv
data/        downloaded splits (gitignored)
results/     generations + metrics per run
```

## Running on Colab

Open `notebooks/colab_baselines.ipynb` in Colab (*File > Open notebook > GitHub*, or upload it),
pick a GPU runtime, add a `GITHUB_TOKEN` secret with read access to this repo, and run all cells.
It runs a 50-question smoke test per model, then the full test set, and saves results to
`MyDrive/implicit-adaptation/`. Finished runs are skipped, so re-running after a disconnect resumes.

## Quick start (any GPU machine)

```bash
PY=~/miniconda3/envs/ICCV/bin/python    # temporary env: torch 2.7, transformers 4.57
$PY -m pip install -r requirements.txt

$PY -m tasks.gsm8k --out data/gsm8k     # train (7,273) / dev (200) / test (1,319)
$PY -m evaluation.generate --model qwen2.5-0.5b-base --few_shot 8 \
    --out results/qwen2.5-0.5b-base/gsm8k/8shot
$PY -m evaluation.generate --model olmo2-1b-sft --out results/olmo2-1b-sft/gsm8k/0shot
$PY -m analysis.summarize results       # one table over all runs
```

Add `--limit 50` to `evaluation.generate` for a quick check. `--dtype auto` uses bf16 on
L4/A100 and float32 on older GPUs (T4, V100).

## Conventions

- Base models: `Question: {q}\nAnswer:`, generation cut at a blank line. Post-trained models:
  their chat template, with the question plus "Solve the problem step by step. End your
  response with "The answer is N."", stopping at EOS only. Reference solutions are the GSM8K
  rationale on one line, without `<<calc>>` notes, ending in `Therefore, the answer is N.`
- Two accuracies: **strict** (`the answer is N`) and **lenient** (first `answer ...`,
  `#### N` or `\boxed{N}`, else the last number). The gap shows how much a model loses to format alone.
- Numbers are normalized before comparing (`$`, commas, trailing `.0`).
- Raw generations are saved, so a parser fix means re-scoring, not re-generating.
