#!/usr/bin/env bash
# RFT baseline for the Base + zero-shot CoT fork-LoRA: sample 8 solutions on the same 1,000 training questions
# (HF sampling, robust to vLLM start-up failures), fine-tune LoRA on correct (gold) / majority-agreeing samples,
# evaluate merged on dev and test500 with the same decoder as the fork-LoRA and the 37.0% / 41.5% baselines.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HPY=${HPY:-$HOME/miniconda3/envs/ICCV/bin/python}; OUT=${OUT:-/data/azfarm/siddhant_fork_lora}; R=results/rft_base; L=logs/queue
COT=" Let's think step by step."
GENS=results/screen/olmo2-1b-base/train/sample_prefix-let-s-think-step-by-step_T0.7_n8
mkdir -p $R $L
GPU=${GPU:?set GPU=<one gpu id>}   # every step runs on this one GPU; waits for its memory instead of using another card
pick_gpu () { while true; do f=$(nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '); \
  [ "$f" -ge "$1" ] && { echo "$GPU"; return; }; sleep 120; done; }
acc () { grep -oE '"acc_lenient": [0-9.]+' $1 | tail -1; }
for try in 1 2 3; do
  [ -f $GENS/train.generations.jsonl ] && break
  g=$(pick_gpu 6000); echo "=== [gpu$g] sample base+CoT n=8 on 1,000 train questions (try $try) $(date)"
  CUDA_VISIBLE_DEVICES=$g $HPY -m methods.run --method sample --model olmo2-1b-base --split train --limit 1000 --n 8 \
    --temperature 0.7 --prefix "$COT" --max_new_tokens 320 --batch_problems 8 > $L/rft_sample.log 2>&1
done
[ -f $GENS/train.generations.jsonl ] || { echo "sampling failed"; exit 1; }
for lab in gold majority; do
  tag=base_rft_$lab
  g=$(pick_gpu 7000); echo "=== [gpu$g] train $tag $(date)"
  CUDA_VISIBLE_DEVICES=$g $HPY -m train.rft_lora --gens $GENS --split train --model olmo2-1b-base --prefix "$COT" \
    --label $lab --lr 1e-4 --epochs 2 --out $OUT/$tag > $L/train_$tag.log 2>&1
  grep -E "training solutions" $L/train_$tag.log | tail -1
  mkdir -p $R/$tag; cp $OUT/$tag/rft_config.json $R/$tag/ 2>/dev/null
  for split in dev test500; do
    g=$(pick_gpu 6000)
    CUDA_VISIBLE_DEVICES=$g $HPY -m methods.run --method greedy --model $OUT/$tag --prefix "$COT" --split $split \
      --max_new_tokens 320 --batch_problems 1 --out $R/$tag/merged_$split > $L/eval_${tag}_$split.log 2>&1
    echo "  $tag $split $(acc $L/eval_${tag}_$split.log)"
  done
done
git add $R train/rft_lora.py scripts/queue_rft_baseline.sh 2>/dev/null
git commit -q -m "RFT baseline (fine-tune on own correct / majority samples) for Base + CoT: dev + test500

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && { [ "${PUSH:-0}" = 1 ] && git push -q 2>&1 | tail -1; true; }
echo "=== done $(date)"
