#!/usr/bin/env bash
# Decide the dev-selected extrapolation (DPO->Instruct, alpha 1.5) on the FULL test (1,319) vs alpha 1 (= Instruct),
# same pipeline. Retries each alpha until it succeeds (shared GPUs: memory can change between pick and eval).
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HPY=$HOME/miniconda3/envs/ICCV/bin/python
pick_gpu () { while true; do g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
  awk -F', ' -v need="$1" '$2 >= need {print $1; exit}'); [ -n "$g" ] && { echo "$g"; return; }; sleep 60; done; }
for a in 1.5 1; do
  for try in 1 2 3 4 5 6; do
    [ -f results/weight_sweep/dpo__instruct_fulltest_a$a/alpha_$a/test.metrics.json ] && break
    g=$(pick_gpu 10000); echo "=== [gpu$g] alpha $a full test, try $try $(date)"
    $HPY -m analysis.weight_sweep --gpu $g --a olmo2-1b-dpo --b olmo2-1b-instruct --alphas $a --split test \
      --tag dpo__instruct_fulltest_a$a --scratch /data/azfarm/siddhant_fork_lora/ws_tmp5 2>&1 | grep -E "^alpha=" | cut -c1-80
    sleep 120
  done
done
echo "=== done $(date)"
