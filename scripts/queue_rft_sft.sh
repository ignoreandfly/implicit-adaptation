#!/usr/bin/env bash
# RFT on OLMo 2 1B SFT (gold and majority/label-free), reusing the judge's training samples
# (8 SFT samples on 1,500 GSM8K training questions). Eval: dev + test500, same HF decoder as the SFT control.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HPY=$HOME/miniconda3/envs/ICCV/bin/python; OUT=/data/azfarm/siddhant_fork_lora; R=results/rft_sft; L=logs/queue
GENS=results/judge/olmo2-1b-sft_train1500_n8
pick_gpu () { while true; do g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
  awk -F', ' -v need="$1" '$2 >= need {print $1; exit}'); [ -n "$g" ] && { echo "$g"; return; }; sleep 90; done; }
until [ -f $GENS/train.generations.jsonl ]; do sleep 120; done
for lab in gold majority; do
  tag=sft_rft_$lab
  g=$(pick_gpu 8000); echo "=== [gpu$g] train $tag $(date)"
  CUDA_VISIBLE_DEVICES=$g $HPY -m train.rft_lora --gens $GENS --split train --model olmo2-1b-sft --label $lab \
    --lr 1e-4 --epochs 2 --out $OUT/$tag > $L/train_$tag.log 2>&1
  grep -E "training solutions" $L/train_$tag.log | tail -1
  mkdir -p $R/$tag; cp $OUT/$tag/rft_config.json $R/$tag/ 2>/dev/null
  for split in dev test500; do
    g=$(pick_gpu 6000)
    CUDA_VISIBLE_DEVICES=$g $HPY -m methods.run --method rl_rules --rl_beta 0 --model $OUT/$tag --format chat --split $split \
      --max_new_tokens 512 --batch_problems 2 --out $R/$tag/merged_$split > $L/eval_${tag}_$split.log 2>&1
    echo "  $tag $split $(grep -oE '"acc_lenient": [0-9.]+' $L/eval_${tag}_$split.log | tail -1)"
  done
done
# SFT control on test500 through the same decoder (dev control = 0.720)
g=$(pick_gpu 6000)
CUDA_VISIBLE_DEVICES=$g $HPY -m methods.run --method rl_rules --rl_beta 0 --model olmo2-1b-sft --split test500 --max_new_tokens 512 \
  --batch_problems 2 --out $R/sft_control/test500 > $L/eval_sft_control_test500.log 2>&1
echo "  sft control test500 $(grep -oE '"acc_lenient": [0-9.]+' $L/eval_sft_control_test500.log | tail -1)"
git add $R scripts/queue_rft_sft.sh 2>/dev/null
git commit -q -m "RFT on 1B SFT (gold / majority): dev + test500

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q 2>&1 | tail -1
echo "=== done $(date)"
