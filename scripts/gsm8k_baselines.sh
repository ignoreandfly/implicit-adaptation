#!/usr/bin/env bash
# Phase 1 GSM8K baselines on a local GPU (same runs and layout as notebooks/colab_baselines.ipynb).
#   GPU=1 bash scripts/gsm8k_baselines.sh            # full test set
#   GPU=1 LIMIT=50 ROOT=results/smoke bash scripts/gsm8k_baselines.sh
# Finished runs (test.metrics.json present) are skipped, so re-running resumes.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-$HOME/miniconda3/envs/ICCV/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=${ROOT:-results}
LIM=${LIMIT:+--limit $LIMIT}
BS_ZERO=${BS_ZERO:-64}
BS_FEW=${BS_FEW:-16}   # 8-shot prompts are ~10x longer

[ -f data/gsm8k/test.jsonl ] || $PY -m tasks.gsm8k --out data/gsm8k

fail=0
for spec in olmo2-1b-base:0 olmo2-1b-base:8 olmo2-1b-sft:0 olmo2-1b-dpo:0 olmo2-1b-instruct:0; do
  model=${spec%:*}; shots=${spec#*:}
  out=$ROOT/$model/gsm8k/${shots}shot
  if [ -f "$out/test.metrics.json" ]; then echo "done already: $out"; continue; fi
  bs=$([ "$shots" = 0 ] && echo $BS_ZERO || echo $BS_FEW)
  echo "=== $model ${shots}-shot  $(date)"
  $PY -m evaluation.generate --model $model --few_shot $shots --batch_size $bs $LIM --out "$out" || fail=1
done

$PY -m analysis.summarize "$ROOT"
exit $fail
