#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export HF_HUB_CACHE=/data/azfarm/siddhant_hf_cache CUDA_VISIBLE_DEVICES=${GPU:-2}
$HOME/miniconda3/envs/implicit-vllm/bin/python -m evaluation.generate --backend vllm --model olmo2-7b-sft --subset 500 \
  --n_samples 64 --temperature 0.7 --gpu_memory_utilization 0.75 --out results/olmo2-7b-sft/gsm8k/0shot_sub500_n64_t0.7
