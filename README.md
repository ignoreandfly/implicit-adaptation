# Implicit Adaptation below 3B Parameters

COMS 4705 final project. Current plan and task list: [ROADMAP.md](ROADMAP.md).

Original question: does Hewitt et al.'s "instruction following without instruction
tuning" hold below 3B parameters, and can the same lightweight interventions bring out
step-by-step reasoning?

## Layout

```
common/      prompt templates, jsonl I/O
tasks/       task generators/loaders + verifiers (arithmetic, gsm8k; last_letter, coin_flip TODO)
train/       lora_sft.py (conditions: full | response_only | corrupted), configs/*.yaml
evaluation/  generate.py (HF generate + scoring), metrics.py (pass@k, format rate, step acc)
adapters/    rule-based logits processors (TODO: Hewitt rules, reasoning rules)
analysis/    summarize.py -> results/summary.csv
scripts/     end-to-end experiment scripts
data/        generated splits            (gitignored)
runs/        LoRA adapters + train logs  (gitignored)
results/     generations + metrics per run
```

## Quick start

```bash
PY=~/miniconda3/envs/ICCV/bin/python          # temporary env: torch 2.7, transformers 4.57, peft 0.17
$PY -m tasks.arithmetic --out data/arithmetic  # 5k train / 200 dev / 500 test_id (2-4 steps) / 500 test_ood (5-6 steps)
GPU=1 SEED=0 bash scripts/arith_qwen0.5b.sh    # base + 3 LoRA conditions, eval, summary table
$PY -m tasks.gsm8k --out data/gsm8k            # needs `datasets`; train / dev (200) / test (1,319)
```

## Conventions

- Prompt: `Question: {q}\nAnswer:` and completion ` {solution}<eos>`. Response-only keeps `Answer:` but drops the question.
- Loss only on completion tokens. LoRA rank, lr, epochs and batch size are fixed per config across conditions.
- Two accuracies: **strict** (`the answer is N`) and **lenient** (last integer). The gap between them shows how much of a gain comes from formatting alone.
- Raw generations are saved, so a parser fix means re-scoring, not re-generating.
