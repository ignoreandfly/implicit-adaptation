#!/usr/bin/env bash
# First experiment: Qwen2.5-0.5B on synthetic arithmetic.
# Base (zero-shot, 3-shot) vs LoRA {full, response_only, corrupted}, greedy pass@1 on test_id + test_ood.
#   GPU=1 SEED=0 bash scripts/arith_qwen0.5b.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-$HOME/miniconda3/envs/ICCV/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-1}
SEED=${SEED:-0}
CFG=train/configs/qwen2.5-0.5b.yaml
MODEL=Qwen/Qwen2.5-0.5B
NAME=qwen2.5-0.5b
RES=results/$NAME/arithmetic

[ -f data/arithmetic/train.jsonl ] || $PY -m tasks.arithmetic --out data/arithmetic --seed 0

evalrun () {  # $1 = out dir, rest = extra args
  local out=$1; shift
  for split in test_id test_ood; do
    [ -f "$out/$split.metrics.json" ] && continue
    $PY -m evaluation.generate --model $MODEL --task arithmetic --split $split --out "$out" "$@"
  done
}

if [ "$SEED" = 0 ]; then
  evalrun $RES/base_zero_shot
  evalrun $RES/base_3shot --few_shot 3
fi

for cond in full response_only corrupted; do
  run=runs/${NAME}_arithmetic_${cond}_s${SEED}
  [ -f $run/adapter/adapter_config.json ] || \
    $PY -m train.lora_sft --config $CFG --task arithmetic --condition $cond --seed $SEED --out $run
  evalrun $RES/${cond}_s${SEED} --adapter $run/adapter
done

$PY -m analysis.summarize results
