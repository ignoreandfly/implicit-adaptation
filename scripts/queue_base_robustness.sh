#!/usr/bin/env bash
# Robustness checks for the gold-label Base + CoT fork-LoRA (+3.2 on test500, p = 0.17):
# shuffled control on test500, a second training seed, and full-test (1,319) evals vs the baseline.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HPY=$HOME/miniconda3/envs/ICCV/bin/python; OUT=/data/azfarm/siddhant_fork_lora; R=results/fork_lora_base; L=logs/queue
COT=" Let's think step by step."; PAIRS=data/fork_pairs/olmo2-1b-base_cot_train1000.jsonl
pick_gpu () { while true; do g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
  awk -F', ' -v need="$1" '$2 >= need {print $1; exit}'); [ -n "$g" ] && { echo "$g"; return; }; sleep 120; done; }
acc () { grep -oE '"acc_lenient": [0-9.]+' $1 | tail -1; }
ev () {  # model, split, outdir, log
  for try in 1 2 3; do
    g=$(pick_gpu 6000)
    CUDA_VISIBLE_DEVICES=$g $HPY -m methods.run --method greedy --model $1 --prefix "$COT" --split $2 --max_new_tokens 320 \
      --batch_problems 1 --out $3 > $4 2>&1
    r=$(acc $4); [ -n "$r" ] && { echo "$r"; return; }
  done; echo "FAILED"
}
echo "=== shuffled test500 $(date)";  echo "  $(ev $OUT/base_dpo_shuffled test500 $R/base_dpo_shuffled/merged_test500 $L/test500_base_shuffled.log)  (baseline 0.370, gold 0.402)"
echo "=== gold seed 1: train $(date)"
g=$(pick_gpu 7000)
CUDA_VISIBLE_DEVICES=$g $HPY -m train.fork_lora --pairs $PAIRS --model olmo2-1b-base --loss dpo --lr 1e-4 --beta 0.5 --epochs 2 \
  --label gold --seed 1 --out $OUT/base_dpo_gold_s1 > $L/train_base_dpo_gold_s1.log 2>&1
mkdir -p $R/base_dpo_gold_s1; cp $OUT/base_dpo_gold_s1/fork_lora_config.json $R/base_dpo_gold_s1/ 2>/dev/null
echo "  gold seed 1 test500 $(ev $OUT/base_dpo_gold_s1 test500 $R/base_dpo_gold_s1/merged_test500 $L/test500_base_gold_s1.log)"
echo "=== full test (1,319) $(date)"
echo "  baseline base+CoT test $(ev olmo2-1b-base test $R/base_cot_baseline/full_test $L/fulltest_base_cot.log)"
echo "  gold fork-LoRA    test $(ev $OUT/base_dpo_gold test $R/base_dpo_gold/merged_fulltest $L/fulltest_base_gold.log)"
git add $R scripts/queue_base_robustness.sh 2>/dev/null
git commit -q -m "Base + CoT fork-LoRA robustness: shuffled on test500, gold seed 1, full-test evals

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q 2>&1 | tail -1
echo "=== done $(date)"
