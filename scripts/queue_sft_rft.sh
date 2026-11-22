#!/usr/bin/env bash
# RFT baseline for the full-data SFT fork-LoRA: 8 chat samples (vLLM, T = 0.7) on all 7,273 training questions,
# LoRA fine-tune on own gold-correct / majority-agreeing solutions (1 epoch, as the full-data fork-LoRA), then
# greedy dev with the same pipeline as results/fork_lora/*/greedy_dev (vLLM, chat, 512 tokens). ONE GPU ($GPU).
set -uo pipefail
AFTER=${AFTER:-}   # optional PID to wait for before starting (keeps GPU memory in budget)
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VPY=${VPY:?}; HPY=${HPY:?}; OUT=${OUT:?}; GPU=${GPU:?}
export CUDA_VISIBLE_DEVICES=$GPU
S=results/screen/olmo2-1b-sft/train/vllm_T0.7_n8; G=results/screen/olmo2-1b-sft/train/vllm_T0.7_n8_paths
R=results/rft_sft; L=logs/queue
mkdir -p $R $L $G
[ -n "$AFTER" ] && while kill -0 $AFTER 2>/dev/null; do sleep 60; done
wait_mem () { while true; do f=$(nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  [ "$f" -ge "$1" ] && return; sleep 60; done; }
for try in $(seq 1 8); do
  [ -f $S/train.generations.jsonl ] && break
  [ $try -gt 1 ] && sleep 300
  wait_mem 16000; echo "=== sample SFT n=8 on 7,273 train questions (try $try) $(date)"
  $VPY -m evaluation.generate --backend vllm --model olmo2-1b-sft --split train --n_samples 8 --temperature 0.7 \
    --max_new_tokens 512 --gpu_memory_utilization 0.3 --out $S > $L/sft_rft_sample.log 2>&1
done
[ -f $S/train.generations.jsonl ] || { echo "sampling failed"; exit 1; }
$HPY scripts/gens_to_paths.py $S/train.generations.jsonl $G/train.generations.jsonl
for lab in gold majority; do
  tag=sft_rft_$lab; rtag=sft_rft_full7273_$lab   # results dir distinct from the 1,500-question run
  wait_mem 12000; echo "=== train $tag $(date)"
  $HPY -m train.rft_lora --gens $G --split train --model olmo2-1b-sft --label $lab --lr 1e-4 --epochs 1 \
    --out $OUT/$tag > $L/train_$tag.log 2>&1
  grep -E "training solutions" $L/train_$tag.log | tail -1
  mkdir -p $R/$rtag; cp $OUT/$tag/rft_config.json $R/$rtag/ 2>/dev/null
  for try in 1 2 3; do
    wait_mem 16000
    $VPY -m evaluation.generate --backend vllm --model $OUT/$tag --format chat --split dev --gpu_memory_utilization 0.3 \
      --out $R/$rtag/greedy_dev > $L/eval_${tag}_dev.log 2>&1 && break; sleep 120
  done
  echo "  $tag dev $(grep -oE "'acc_lenient': [0-9.]+" $L/eval_${tag}_dev.log | tail -1)"
done
git add $R scripts/queue_sft_rft.sh scripts/gens_to_paths.py 2>/dev/null
git commit -q -m "RFT baseline on 1B SFT, full training set (gold / majority), dev

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
echo "=== done $(date)"
