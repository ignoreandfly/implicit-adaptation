#!/usr/bin/env bash
# Base + zero-shot CoT, fork-LoRA vs RFT (same 1,000 training questions), all on ONE GPU ($GPU):
#   lane 1  rebuild fork-LoRA gold (pairs + train, same config as results/fork_lora_base/base_dpo_gold),
#           reproduce its test500 greedy (40.2%), then fork-DPO on top of RFT gold (on-policy pairs from the
#           RFT model) -> greedy dev + test500
#   lane 2  diversity: n = 16 samples (T = 0.7) on test500 for Base, RFT gold, fork-LoRA gold, RFT+fork-DPO
#           -> acc_single_sample (pass@1), acc_majority (maj@16), acc_oracle (pass@16)
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VPY=${VPY:?}; HPY=${HPY:?}; OUT=${OUT:?}; GPU=${GPU:?}
export CUDA_VISIBLE_DEVICES=$GPU
COT=" Let's think step by step."
R=results/fork_lora_base; D=results/diversity; L=logs/queue
mkdir -p $R $D $L data/fork_pairs
wait_mem () { while true; do f=$(nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  [ "$f" -ge "$1" ] && return; sleep 60; done; }
acc () { grep -oE '"acc_lenient": [0-9.]+' $1 | tail -1; }
pairs () {  # model, out
  for try in $(seq 1 8); do
    [ -f $2 ] && return 0
    [ $try -gt 1 ] && sleep 300
    wait_mem 16000; echo "=== pairs $1 (try $try) $(date)"
    $VPY -m train.fork_pairs --model $1 --prefix "$COT" --n 1000 --max_tokens 320 --gpu_memory_utilization 0.3 \
      --out $2 > $L/pairs_$(basename $2 .jsonl).log 2>&1; tail -1 $L/pairs_$(basename $2 .jsonl).log
  done
  [ -f $2 ]
}
greedy () {  # model, split, outdir, tag
  wait_mem 8000
  $HPY -m methods.run --method greedy --model $1 --prefix "$COT" --split $2 --max_new_tokens 320 --batch_problems 1 \
    --out $3 > $L/eval_$4_$2.log 2>&1; echo "  $4 $2 greedy $(acc $L/eval_$4_$2.log)"
}
sample16 () {  # model, tag
  wait_mem 12000; echo "=== sample n=16 $2 $(date)"
  $HPY -m methods.run --method sample --model $1 --prefix "$COT" --split test500 --n 16 --temperature 0.7 \
    --max_new_tokens 320 --batch_problems 8 --out $D/$2 > $L/div_$2.log 2>&1
  python3 -c "import json;m=json.load(open('$D/$2/test500.metrics.json'));print('  $2', {k:round(m[k],3) for k in ('acc_single_sample','acc_majority','acc_oracle')})" 2>&1
}

lane1 () {
  P0=data/fork_pairs/olmo2-1b-base_cot_train1000.jsonl
  pairs olmo2-1b-base $P0 || { echo "lane1: no base pairs"; return 1; }
  wait_mem 12000; echo "=== train base_dpo_gold (rebuild) $(date)"
  $HPY -m train.fork_lora --pairs $P0 --model olmo2-1b-base --loss dpo --lr 1e-4 --beta 0.5 --epochs 2 --label gold \
    --out $OUT/base_dpo_gold > $L/train_base_dpo_gold.log 2>&1; grep -E "training forks" $L/train_base_dpo_gold.log | tail -1
  greedy $OUT/base_dpo_gold test500 $R/base_dpo_gold/rebuild_test500 base_dpo_gold_rebuild
  P1=data/fork_pairs/base_rft_gold_cot_train1000.jsonl
  pairs $OUT/base_rft_gold $P1 || { echo "lane1: no RFT pairs"; return 1; }
  wait_mem 12000; echo "=== train base_rft_gold_forkdpo_gold $(date)"
  $HPY -m train.fork_lora --pairs $P1 --model $OUT/base_rft_gold --loss dpo --lr 1e-4 --beta 0.5 --epochs 2 --label gold \
    --out $OUT/base_rft_gold_forkdpo_gold > $L/train_base_rft_gold_forkdpo_gold.log 2>&1
  grep -E "training forks" $L/train_base_rft_gold_forkdpo_gold.log | tail -1
  mkdir -p results/rft_base/base_rft_gold_forkdpo_gold
  cp $OUT/base_rft_gold_forkdpo_gold/fork_lora_config.json results/rft_base/base_rft_gold_forkdpo_gold/ 2>/dev/null
  for s in dev test500; do greedy $OUT/base_rft_gold_forkdpo_gold $s results/rft_base/base_rft_gold_forkdpo_gold/merged_$s rft_forkdpo; done
  sample16 $OUT/base_dpo_gold base_dpo_gold
  sample16 $OUT/base_rft_gold_forkdpo_gold base_rft_gold_forkdpo_gold
}
lane2 () {
  sample16 olmo2-1b-base base_cot
  sample16 $OUT/base_rft_gold base_rft_gold
}

lane1 > $L/forkvsrft_lane1.out 2>&1 &
sleep 240   # let lane 1's vLLM finish start-up profiling before lane 2 changes memory
lane2 > $L/forkvsrft_lane2.out 2>&1 &
wait
cat $L/forkvsrft_lane1.out $L/forkvsrft_lane2.out
git add $R $D results/rft_base scripts/queue_base_forkvsrft.sh 2>/dev/null
git commit -q -m "Base + CoT: fork-LoRA rebuild, RFT + fork-DPO, n=16 diversity (test500)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
echo "=== done $(date)"
