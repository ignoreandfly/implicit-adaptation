#!/usr/bin/env bash
# Decide the dev-selected extrapolation (DPO->Instruct, alpha 1.5) on the FULL test (1,319) vs Instruct, same pipeline.
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HPY=$HOME/miniconda3/envs/ICCV/bin/python
pick_gpu () { while true; do g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
  awk -F', ' -v need="$1" '$2 >= need {print $1; exit}'); [ -n "$g" ] && { echo "$g"; return; }; sleep 60; done; }
g=$(pick_gpu 9000); echo "=== [gpu$g] alpha 1.5 full test $(date)"
$HPY -m analysis.weight_sweep --gpu $g --a olmo2-1b-dpo --b olmo2-1b-instruct --alphas 1.5 1 --split test \
  --tag dpo__instruct_fulltest --scratch /data/azfarm/siddhant_fork_lora/ws_tmp5 2>&1 | grep -E "^alpha="
echo "=== done $(date)"
