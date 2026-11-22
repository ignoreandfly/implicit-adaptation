#!/usr/bin/env bash
# Follow-up: (1) DPO->Instruct extrapolation (dev peak alpha 1.5: 0.825 vs Instruct 0.785) on test500 + finer dev grid;
# (2) rerun the SFT->RFT-gold extrapolation points that OOM'd.
set -uo pipefail
cd "$(dirname "$0")/.."
HPY=$HOME/miniconda3/envs/ICCV/bin/python
pick_gpu () { while true; do g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
  awk -F', ' -v need="$1" '$2 >= need {print $1; exit}'); [ -n "$g" ] && { echo "$g"; return; }; sleep 90; done; }
run () { g=$(pick_gpu 7000); echo "=== [gpu$g] $* $(date)"; PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $HPY -m analysis.weight_sweep --gpu $g "$@" 2>&1 | grep -E "^alpha="; }
run --a olmo2-1b-dpo --b olmo2-1b-instruct --alphas 1.5 1.25 1.75 --split test500 --tag dpo__instruct --scratch /data/azfarm/siddhant_fork_lora/ws_tmp2
run --a olmo2-1b-dpo --b olmo2-1b-instruct --alphas 1.25 1.75 --split dev --tag dpo__instruct_fine --scratch /data/azfarm/siddhant_fork_lora/ws_tmp2
run --a olmo2-1b-sft --b /data/azfarm/siddhant_fork_lora/sft_rft_gold --alphas 1.5 2 3 --tag sft__sft_rft_gold --scratch /data/azfarm/siddhant_fork_lora/ws_tmp3
echo "=== done $(date)"
