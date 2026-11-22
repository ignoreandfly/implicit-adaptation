#!/usr/bin/env bash
# Base + CoT RFT models (gold, majority) on the full GSM8K test (1,319), same decoder as the 38.1% baseline.
set -uo pipefail
cd "$(dirname "$0")/.."
HPY=$HOME/miniconda3/envs/ICCV/bin/python; OUT=/data/azfarm/siddhant_fork_lora; R=results/rft_base; L=logs/queue
export CUDA_VISIBLE_DEVICES=${GPU:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COT=" Let's think step by step."
for tag in base_rft_gold base_rft_majority; do
  $HPY -m methods.run --method greedy --model $OUT/$tag --prefix "$COT" --split test --max_new_tokens 320 --batch_problems 1 \
    --out $R/$tag/merged_fulltest > $L/fulltest_$tag.log 2>&1
  echo "$tag full test $(grep -oE '"acc_lenient": [0-9.]+' $L/fulltest_$tag.log | tail -1)   (baseline 0.381)"
done
git add $R scripts/rft_base_fulltest.sh 2>/dev/null
git commit -q -m "Base + CoT RFT (gold / majority) on the full GSM8K test

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q 2>&1 | tail -1
