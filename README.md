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

| Family | Checkpoint | Post-training |
|---|---|---|
| OLMo 2 1B | `allenai/OLMo-2-0425-1B` | none |
| OLMo 2 1B | `allenai/OLMo-2-0425-1B-SFT` | SFT (Tülu 3 mix, includes math CoT) |
| OLMo 2 1B | `allenai/OLMo-2-0425-1B-DPO` | SFT + DPO |
| OLMo 2 1B | `allenai/OLMo-2-0425-1B-Instruct` | SFT + DPO + RLVR (math) |
| Qwen2.5 | `Qwen/Qwen2.5-0.5B` | none |
| Qwen2.5 | `Qwen/Qwen2.5-0.5B-Instruct` | SFT + DPO + GRPO (learned reward model) |

## Layout

```
common/      prompt templates, jsonl I/O
tasks/       gsm8k.py: loader (writes data/gsm8k/*.jsonl) + answer verifier
evaluation/  generate.py (HF generate + scoring), metrics.py (accuracy, pass@k, format rate, step acc)
adapters/    rule-based logits processors (TODO: Hewitt adapter)
analysis/    summarize.py -> results/summary.csv
data/        downloaded splits (gitignored)
results/     generations + metrics per run
```

## Quick start

```bash
PY=~/miniconda3/envs/ICCV/bin/python    # temporary env: torch 2.7, transformers 4.57
$PY -m pip install -r requirements.txt

$PY -m tasks.gsm8k --out data/gsm8k     # train (7,273) / dev (200) / test (1,319)
$PY -m evaluation.generate --model Qwen/Qwen2.5-0.5B --split test --few_shot 8 \
    --out results/qwen2.5-0.5b/gsm8k/base_8shot
$PY -m analysis.summarize results       # one table over all runs
```

Add `--limit 50` to `evaluation.generate` for a quick check. Chat templates for the
post-trained models are not wired in yet (see ROADMAP.md, Phase 1 setup).

## Conventions

- Base-model prompt: `Question: {q}\nAnswer:`. Reference solutions are the GSM8K
  rationale on one line, without `<<calc>>` notes, ending in `Therefore, the answer is N.`
- Two accuracies: **strict** (`the answer is N`) and **lenient** (first `answer ...` /
  `#### N` phrase, else the last number). The gap shows how much a model loses to format alone.
- Numbers are normalized before comparing (`$`, commas, trailing `.0`).
- Raw generations are saved, so a parser fix means re-scoring, not re-generating.
