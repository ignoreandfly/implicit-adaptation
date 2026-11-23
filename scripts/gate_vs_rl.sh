#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HPY=$HOME/miniconda3/envs/ICCV/bin/python
for pair in "olmo2-1b-dpo olmo2-1b-instruct olmo2-1b-dpo_train300" "olmo2-1b-sft olmo2-1b-instruct olmo2-1b-sft_train300"; do
  set -- $pair
  echo "=== $1 -> $2 $(date)"
  CUDA_VISIBLE_DEVICES=0 $HPY -m analysis.gate_vs_rl --gens results/rl_delta/$3 --a $1 --b $2 > logs/queue/gate_vs_rl_$1_$2.log 2>&1
  tail -42 logs/queue/gate_vs_rl_$1_$2.log | grep -E "auroc|entropy|margin|jaccard|share|recall" | head -40
done
