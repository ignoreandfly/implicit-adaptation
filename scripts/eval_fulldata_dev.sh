#!/usr/bin/env bash
# Dev evals for the full-data SFT fork-LoRA models (queue_fulldata_retry.sh). That script's own eval step calls
# `methods.run --format chat`, which methods.run does not accept, so all three evals failed; this re-runs them
# with the round-3 pipeline (results/fork_lora/*/greedy_dev): vLLM greedy, chat format, 512 tokens. ONE GPU.
#   GPU=0 VPY=... OUT=/path/to/fork_lora [WAIT_PID=<pid of queue_fulldata_retry.sh>] bash scripts/eval_fulldata_dev.sh
set -uo pipefail
cd "$(dirname "$0")/.."
VPY=${VPY:?}; OUT=${OUT:?}; GPU=${GPU:?}; R=results/fork_lora; L=logs/queue
export CUDA_VISIBLE_DEVICES=$GPU
mkdir -p $L
[ -n "${WAIT_PID:-}" ] && while kill -0 $WAIT_PID 2>/dev/null; do sleep 60; done
for tag in full_dpo_major_lr1e-4 full_dpo_gold_lr1e-4 full_dpo_shuffled_lr1e-4; do
  for try in 1 2 3; do
    $VPY -m evaluation.generate --backend vllm --model $OUT/$tag --format chat --split dev --gpu_memory_utilization 0.3 \
      --out $R/$tag/greedy_dev > $L/eval_${tag}_vllm.log 2>&1 && break; sleep 120
  done
  echo "$tag dev $(grep -oE "'acc_lenient': [0-9.]+" $L/eval_${tag}_vllm.log | tail -1)"
done
echo "=== evals done $(date)"
