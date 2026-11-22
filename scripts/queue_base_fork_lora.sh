#!/usr/bin/env bash
# Lane C: fork-contrastive self-distillation on OLMo 2 1B BASE with the zero-shot CoT prefix.
# Each job waits until a GPU has enough free memory. Data on 1,000 training questions -> LoRA (majority /
# gold / shuffled; bounded DPO loss, lr 1e-4, 2 epochs) -> dev eval merged AND fork-only, vs. Base + CoT.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VPY=$HOME/miniconda3/envs/implicit-vllm/bin/python; HPY=$HOME/miniconda3/envs/ICCV/bin/python
COT=" Let's think step by step."
PAIRS=data/fork_pairs/olmo2-1b-base_cot_train1000.jsonl
OUT=/data/azfarm/siddhant_fork_lora; R=results/fork_lora_base; L=logs/queue
mkdir -p $R $L data/fork_pairs
pick_gpu () { while true; do g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
  awk -F', ' -v need="$1" '$2 >= need {print $1; exit}'); [ -n "$g" ] && { echo "$g"; return; }; sleep 120; done; }
acc () { grep -oE '"acc_lenient": [0-9.]+' $1 | tail -1; }

# baseline: Base + zero-shot CoT, greedy, dev (same decoder as the evals below)
g=$(pick_gpu 9000); echo "=== [C gpu$g] baseline base+CoT dev $(date)"
CUDA_VISIBLE_DEVICES=$g $HPY -m methods.run --method greedy --model olmo2-1b-base --prefix "$COT" --split dev \
  --max_new_tokens 320 --out $R/base_cot_baseline > $L/base_baseline.log 2>&1; echo "  $(acc $L/base_baseline.log)"

# data (vLLM, retry up to 5 times if the engine cannot start because the GPU filled up meanwhile)
for try in 1 2 3 4 5; do
  [ -f $PAIRS ] && break
  g=$(pick_gpu 9000); echo "=== [C gpu$g] base fork pairs (try $try) $(date)"
  CUDA_VISIBLE_DEVICES=$g $VPY -m train.fork_pairs --model olmo2-1b-base --prefix "$COT" --n 1000 --max_tokens 320 \
    --gpu_memory_utilization 0.3 --out $PAIRS > $L/base_fork_pairs.log 2>&1; tail -1 $L/base_fork_pairs.log
done

for spec in "base_dpo_major:--label majority" "base_dpo_gold:--label gold" "base_dpo_shuffled:--label majority --shuffle"; do
  tag=${spec%%:*}; args=${spec#*:}
  g=$(pick_gpu 10000); echo "=== [C gpu$g] train $tag $(date)"
  CUDA_VISIBLE_DEVICES=$g $HPY -m train.fork_lora --pairs $PAIRS --model olmo2-1b-base --loss dpo --lr 1e-4 --beta 0.5 \
    --epochs 2 $args --out $OUT/$tag > $L/train_$tag.log 2>&1
  grep -E "training forks" $L/train_$tag.log | tail -1
  mkdir -p $R/$tag; cp $OUT/$tag/fork_lora_config.json $R/$tag/ 2>/dev/null
  g=$(pick_gpu 9000); echo "=== [C gpu$g] eval $tag merged + fork-only $(date)"
  CUDA_VISIBLE_DEVICES=$g $HPY -m methods.run --method greedy --model $OUT/$tag --prefix "$COT" --split dev \
    --max_new_tokens 320 --out $R/$tag/merged_dev > $L/eval_merged_$tag.log 2>&1; echo "  merged    $(acc $L/eval_merged_$tag.log)"
  CUDA_VISIBLE_DEVICES=$g $HPY -m methods.run --method rl_rules --model olmo2-1b-base --rl_oracle --rl_oracle_model $OUT/$tag \
    --rl_fork_thr 1.0 --prefix "$COT" --split dev --max_new_tokens 320 --batch_problems 4 \
    --out $R/$tag/forkonly_dev > $L/eval_forkonly_$tag.log 2>&1; echo "  fork-only $(acc $L/eval_forkonly_$tag.log)"
done
git add $R scripts/queue_base_fork_lora.sh train/fork_pairs.py methods/rl_rules.py 2>/dev/null
git commit -q -m "Fork-LoRA on 1B Base + zero-shot CoT (dev): majority / gold / shuffled, merged and fork-only

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q 2>&1 | tail -1
echo "=== lane C done $(date)"
