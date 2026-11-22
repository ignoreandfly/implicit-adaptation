#!/usr/bin/env bash
# Sanity check for the fork-LoRA pipeline: lr = 0 (no learning) -> merge -> save -> vLLM eval on dev.
# Should reproduce the SFT baseline dev greedy (0.695); if not, the pipeline itself is damaging the model.
set -uo pipefail
cd "$(dirname "$0")/.."
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
export CUDA_VISIBLE_DEVICES=${GPU:-0}
OUT=/data/azfarm/siddhant_fork_lora/sanity_lr0
$HOME/miniconda3/envs/ICCV/bin/python -m train.fork_lora --pairs data/fork_pairs/olmo2-1b-sft_train1000.jsonl \
  --model olmo2-1b-sft --label majority --lr 0 --epochs 1 --out $OUT 2>&1 | grep -E "saved|Error" | tail -1
$HOME/miniconda3/envs/implicit-vllm/bin/python -m evaluation.generate --backend vllm --model $OUT --format chat --split dev \
  --gpu_memory_utilization 0.5 --out results/fork_lora/sanity_lr0/greedy_dev 2>&1 | grep -oE "'acc_lenient': [0-9.]+" | sed "s/^/  sanity lr=0 dev greedy (baseline 0.695): /"
