#!/usr/bin/env bash
# One robustness job for the Base + CoT fork-LoRA, run in parallel with the others. Usage:
#   robust_job.sh seed1 | full_base | full_gold
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HPY=$HOME/miniconda3/envs/ICCV/bin/python; OUT=/data/azfarm/siddhant_fork_lora; R=results/fork_lora_base; L=logs/queue
COT=" Let's think step by step."; PAIRS=data/fork_pairs/olmo2-1b-base_cot_train1000.jsonl
pick_gpu () { while true; do g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
  awk -F', ' -v need="$1" '$2 >= need {print $1; exit}'); [ -n "$g" ] && { echo "$g"; return; }; sleep 90; done; }
ev () { for try in 1 2 3; do g=$(pick_gpu 5000)
    CUDA_VISIBLE_DEVICES=$g $HPY -m methods.run --method greedy --model $1 --prefix "$COT" --split $2 --max_new_tokens 320 \
      --batch_problems 1 --out $3 > $4 2>&1
    r=$(grep -oE '"acc_lenient": [0-9.]+' $4 | tail -1); [ -n "$r" ] && { echo "$r"; return; }; done; echo FAILED; }
case $1 in
  seed1) g=$(pick_gpu 7000)
         CUDA_VISIBLE_DEVICES=$g $HPY -m train.fork_lora --pairs $PAIRS --model olmo2-1b-base --loss dpo --lr 1e-4 --beta 0.5 \
           --epochs 2 --label gold --seed 1 --out $OUT/base_dpo_gold_s1 > $L/train_base_dpo_gold_s1.log 2>&1
         mkdir -p $R/base_dpo_gold_s1; cp $OUT/base_dpo_gold_s1/fork_lora_config.json $R/base_dpo_gold_s1/ 2>/dev/null
         echo "gold seed 1 test500 $(ev $OUT/base_dpo_gold_s1 test500 $R/base_dpo_gold_s1/merged_test500 $L/test500_base_gold_s1.log)" ;;
  full_base) echo "base+CoT full test $(ev olmo2-1b-base test $R/base_cot_baseline/full_test $L/fulltest_base_cot.log)" ;;
  full_gold) echo "gold fork-LoRA full test $(ev $OUT/base_dpo_gold test $R/base_dpo_gold/merged_fulltest $L/fulltest_base_gold.log)" ;;
esac
git add $R 2>/dev/null; git commit -q -m "Base fork-LoRA robustness: $1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q 2>&1 | tail -1
