#!/usr/bin/env bash
# Base + zero-shot CoT fork-LoRA: decisive test-set check (test500, same decoder as the 37.0% baseline),
# plus a retry of the gold model's fork-only dev eval (failed for memory). Commits + pushes at the end.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HPY=$HOME/miniconda3/envs/ICCV/bin/python; OUT=/data/azfarm/siddhant_fork_lora; R=results/fork_lora_base; L=logs/queue
COT=" Let's think step by step."
pick_gpu () { while true; do g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
  awk -F', ' -v need="$1" '$2 >= need {print $1; exit}'); [ -n "$g" ] && { echo "$g"; return; }; sleep 120; done; }
acc () { grep -oE '"acc_lenient": [0-9.]+' $1 | tail -1; }
for tag in base_dpo_gold base_dpo_major; do
  for try in 1 2 3; do
    g=$(pick_gpu 6000); echo "=== [gpu$g] $tag test500 merged (try $try) $(date)"
    CUDA_VISIBLE_DEVICES=$g $HPY -m methods.run --method greedy --model $OUT/$tag --prefix "$COT" --split test500 \
      --max_new_tokens 320 --batch_problems 1 --out $R/$tag/merged_test500 > $L/test500_$tag.log 2>&1
    r=$(acc $L/test500_$tag.log); [ -n "$r" ] && { echo "  $r   (Base+CoT test500 baseline 0.370)"; break; }
  done
done
for try in 1 2 3; do
  g=$(pick_gpu 8000); echo "=== [gpu$g] base_dpo_gold fork-only dev (try $try) $(date)"
  CUDA_VISIBLE_DEVICES=$g $HPY -m methods.run --method rl_rules --model olmo2-1b-base --rl_oracle --rl_oracle_model $OUT/base_dpo_gold \
    --rl_fork_thr 1.0 --prefix "$COT" --split dev --max_new_tokens 320 --batch_problems 1 \
    --out $R/base_dpo_gold/forkonly_dev > $L/forkonly_base_gold.log 2>&1
  r=$(acc $L/forkonly_base_gold.log); [ -n "$r" ] && { echo "  $r   (Base+CoT dev baseline 0.415)"; break; }
done
git add $R scripts/queue_base_test500.sh 2>/dev/null
git commit -q -m "Base + CoT fork-LoRA on test500 (gold / majority) and gold fork-only dev retry

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q 2>&1 | tail -1
echo "=== done $(date)"
