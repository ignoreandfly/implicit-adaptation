#!/usr/bin/env bash
# Re-run the dev evals of the full-data SFT fork-LoRA models (the first attempt passed --format, which
# methods.run did not accept). Waits for the shuffled model's training to finish (by file).
set -uo pipefail
cd "$(dirname "$0")/.."
HPY=$HOME/miniconda3/envs/ICCV/bin/python; OUT=/data/azfarm/siddhant_fork_lora; R=results/fork_lora; L=logs/queue
pick_gpu () { while true; do g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
  awk -F', ' -v need="$1" '$2 >= need {print $1; exit}'); [ -n "$g" ] && { echo "$g"; return; }; sleep 90; done; }
until [ -f $OUT/full_dpo_shuffled_lr1e-4/model.safetensors ]; do sleep 120; done
for tag in full_dpo_major_lr1e-4 full_dpo_gold_lr1e-4 full_dpo_shuffled_lr1e-4; do
  g=$(pick_gpu 6000)
  CUDA_VISIBLE_DEVICES=$g $HPY -m methods.run --method rl_rules --rl_beta 0 --model $OUT/$tag --format chat --split dev \
    --max_new_tokens 512 --batch_problems 2 --out $R/$tag/merged_dev > $L/eval_$tag.log 2>&1
  echo "$tag dev $(grep -oE '"acc_lenient": [0-9.]+' $L/eval_$tag.log | tail -1)   (SFT control in this decoder: 0.720)"
done
git add $R methods/run.py scripts/eval_fulldata_fork.sh 2>/dev/null
git commit -q -m "Full-data SFT fork-LoRA dev evals (fixed --format for local chat models)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q 2>&1 | tail -1
