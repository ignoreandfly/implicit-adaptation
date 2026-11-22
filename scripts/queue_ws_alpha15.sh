#!/usr/bin/env bash
cd "$(dirname "$0")/.."
while ps -p 809377 >/dev/null; do sleep 60; done
HPY=$HOME/miniconda3/envs/ICCV/bin/python
pick_gpu () { while true; do g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits |   awk -F', ' -v need="$1" '$2 >= need {print $1; exit}'); [ -n "$g" ] && { echo "$g"; return; }; sleep 90; done; }
g=$(pick_gpu 9000); echo "=== [gpu$g] alpha 1.5 test500 $(date)"
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $HPY -m analysis.weight_sweep --gpu $g --a olmo2-1b-dpo --b olmo2-1b-instruct   --alphas 1.5 --split test500 --tag dpo__instruct_a15 --scratch /data/azfarm/siddhant_fork_lora/ws_tmp4 2>&1 | grep -E "^alpha="
