#!/usr/bin/env bash
# Retry of lane A2/A3 (full-data fork pairs on all 7,273 training questions, then SFT fork-LoRA retraining)
# after the verifier crash on a 400-digit "number". Patient vLLM retries; trains only if the data exists.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VPY=${VPY:-$HOME/miniconda3/envs/implicit-vllm/bin/python}; HPY=${HPY:-$HOME/miniconda3/envs/ICCV/bin/python}
PAIRS=data/fork_pairs/olmo2-1b-sft_trainall.jsonl; OUT=${OUT:-/data/azfarm/siddhant_fork_lora}; R=results/fork_lora; L=logs/queue
mkdir -p $L $(dirname $PAIRS)
GPU=${GPU:?set GPU=<one gpu id>}   # every step runs on this one GPU; waits for its memory instead of using another card
pick_gpu () { while true; do f=$(nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '); \
  [ "$f" -ge "$1" ] && { echo "$GPU"; return; }; sleep 120; done; }
for try in $(seq 1 12); do
  [ -f $PAIRS ] && break
  [ $try -gt 1 ] && sleep 600
  g=$(pick_gpu 9000); echo "=== [A2' gpu$g] fork pairs, all training questions (try $try) $(date)"
  CUDA_VISIBLE_DEVICES=$g $VPY -m train.fork_pairs --model olmo2-1b-sft --n 7273 --gpu_memory_utilization 0.3 \
    --out $PAIRS > $L/fork_pairs_all.log 2>&1; tail -1 $L/fork_pairs_all.log
done
[ -f $PAIRS ] || { echo "no full-data pairs; stopping"; exit 1; }
for spec in "full_dpo_major_lr1e-4:--label majority" "full_dpo_gold_lr1e-4:--label gold" "full_dpo_shuffled_lr1e-4:--label majority --shuffle"; do
  tag=${spec%%:*}; args=${spec#*:}
  g=$(pick_gpu 10000); echo "=== [A3' gpu$g] train $tag $(date)"
  CUDA_VISIBLE_DEVICES=$g $HPY -m train.fork_lora --pairs $PAIRS --model olmo2-1b-sft --loss dpo --lr 1e-4 --beta 0.5 \
    --epochs 1 $args --out $OUT/$tag > $L/train_$tag.log 2>&1
  grep -E "training forks" $L/train_$tag.log | tail -1
  mkdir -p $R $L/$tag; cp $OUT/$tag/fork_lora_config.json $R/$tag/ 2>/dev/null
  g=$(pick_gpu 9000)
  CUDA_VISIBLE_DEVICES=$g $HPY -m methods.run --method rl_rules --rl_beta 0 --model $OUT/$tag --format chat --split dev \
    --max_new_tokens 512 --batch_problems 4 --out $R/$tag/merged_dev > $L/eval_$tag.log 2>&1
  echo "  $tag merged dev $(grep -oE '"acc_lenient": [0-9.]+' $L/eval_$tag.log | tail -1)"
done
git add $R scripts/queue_fulldata_retry.sh 2>/dev/null
git commit -q -m "Full-data (7,273 questions) fork-LoRA on 1B SFT: majority / gold / shuffled (dev)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && { [ "${PUSH:-0}" = 1 ] && git push -q 2>&1 | tail -1; true; }
