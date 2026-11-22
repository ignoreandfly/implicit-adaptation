#!/usr/bin/env bash
# OLMo 2 7B SFT on the 500-problem GSM8K test subset: greedy, then n = 64 sampling (T = 0.7) with vLLM.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_CACHE=/data/azfarm/siddhant_hf_cache CUDA_VISIBLE_DEVICES=${GPU:-2}
PY=$HOME/miniconda3/envs/implicit-vllm/bin/python
echo "=== greedy $(date +%T)"
$PY -m evaluation.generate --backend vllm --model olmo2-7b-sft --subset 500 --gpu_memory_utilization 0.8 \
    --out results/olmo2-7b-sft/gsm8k/0shot_sub500_greedy 2>&1 | grep -vE "INFO|WARN" | tail -3
echo "=== n=64 $(date +%T)"
$PY -m evaluation.generate --backend vllm --model olmo2-7b-sft --subset 500 --n_samples 64 --temperature 0.7 \
    --gpu_memory_utilization 0.8 --out results/olmo2-7b-sft/gsm8k/0shot_sub500_n64_t0.7 2>&1 | grep -vE "INFO|WARN" | tail -3
