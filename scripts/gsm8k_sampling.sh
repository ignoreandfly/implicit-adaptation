#!/usr/bin/env bash
# Phase 1 sampling runs (pass@k, maj@k) with the vLLM backend on the fixed 500-problem test subset.
#   GPU=1 bash scripts/gsm8k_sampling.sh                   # n = 64, T = 0.8, 500 problems
#   GPU=1 N=16 SUBSET=500 MEM=0.35 bash scripts/gsm8k_sampling.sh
# Needs the implicit-vllm env (README). Finished runs (test.metrics.json present) are skipped.
set -uo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-$HOME/miniconda3/envs/implicit-vllm/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-1}
N=${N:-64}
TEMP=${TEMP:-0.8}
SUBSET=${SUBSET:-500}
MEM=${MEM:-0.35}        # vLLM gpu_memory_utilization; the GPU is shared
ROOT=${ROOT:-results}
DATA=${DATA:-data/gsm8k}

fail=0
for spec in olmo2-1b-base:0 olmo2-1b-base:8 olmo2-1b-sft:0 olmo2-1b-dpo:0 olmo2-1b-instruct:0; do
  model=${spec%:*}; shots=${spec#*:}
  out=$ROOT/$model/gsm8k/${shots}shot_sub${SUBSET}_n${N}_t${TEMP}
  if [ -f "$out/test.metrics.json" ]; then echo "done already: $out"; continue; fi
  echo "=== $model ${shots}-shot n=$N  $(date)"
  $PY -m evaluation.generate --backend vllm --model $model --few_shot $shots --data_dir $DATA \
      --subset $SUBSET --n_samples $N --temperature $TEMP --gpu_memory_utilization $MEM \
      --out "$out" || fail=1
done

$PY -m analysis.summarize "$ROOT"
exit $fail
