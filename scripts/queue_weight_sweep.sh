#!/usr/bin/env bash
# Training-free weight interpolation / extrapolation along the 1B ladder (dev greedy; SFT 0.720, Instruct 0.790).
#   1. SFT -> DPO, extrapolated past DPO: can extrapolating the DPO step recover what RLVR adds?
#   2. DPO -> Instruct: interpolation and extrapolation past the RL model
#   3. SFT -> our RFT-on-own-correct-samples model, extrapolated: amplify self-training without more training
# Runs one sweep at a time on whichever GPU has >= 7 GB free; each sweep pins its GPU.
set -uo pipefail
cd "$(dirname "$0")/.."
HPY=$HOME/miniconda3/envs/ICCV/bin/python
pick_gpu () { while true; do g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
  awk -F', ' -v need="$1" '$2 >= need {print $1; exit}'); [ -n "$g" ] && { echo "$g"; return; }; sleep 90; done; }
run () { g=$(pick_gpu 7000); echo "=== [gpu$g] $* $(date)"; $HPY -m analysis.weight_sweep --gpu $g "$@" 2>&1 | grep -E "^alpha="; }
run --a olmo2-1b-sft --b olmo2-1b-dpo --alphas 1 1.5 2 3 4 --tag sft__dpo --scratch /data/azfarm/siddhant_fork_lora/ws_tmp1
run --a olmo2-1b-dpo --b olmo2-1b-instruct --alphas 0.5 1 1.5 2 --tag dpo__instruct --scratch /data/azfarm/siddhant_fork_lora/ws_tmp2
run --a olmo2-1b-sft --b /data/azfarm/siddhant_fork_lora/sft_rft_gold --alphas 1 1.5 2 3 --tag sft__sft_rft_gold --scratch /data/azfarm/siddhant_fork_lora/ws_tmp3
git add results/weight_sweep analysis/weight_sweep.py scripts/queue_weight_sweep.sh 2>/dev/null
git commit -q -m "Weight interpolation/extrapolation sweeps along the 1B ladder (dev)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git pull -q --rebase && git push -q 2>&1 | tail -1
echo "=== done $(date)"
