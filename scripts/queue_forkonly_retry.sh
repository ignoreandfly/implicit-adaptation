#!/usr/bin/env bash
# Retry of lane A1 (fork-only inference with round-3 LoRAs on 1B SFT, dev): the first attempt ran out of
# memory when other users' jobs grew on the same GPU. Smaller batch, larger free-memory requirement.
set -uo pipefail
cd "$(dirname "$0")/.."
HPY=$HOME/miniconda3/envs/ICCV/bin/python; OUT=/data/azfarm/siddhant_fork_lora; L=logs/queue
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pick_gpu () { while true; do g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
  awk -F', ' -v need="$1" '$2 >= need {print $1; exit}'); [ -n "$g" ] && { echo "$g"; return; }; sleep 120; done; }
for tag in dpo_major_lr1e-4_e2 dpo_gold_lr1e-4_e2 dpo_shuffled_lr1e-4_e2 dpo_major_lr3e-4_e2; do
  for try in 1 2 3; do
    g=$(pick_gpu 13000); echo "=== [A1' gpu$g] fork-only $tag (try $try) $(date)"
    CUDA_VISIBLE_DEVICES=$g $HPY -m methods.run --method rl_rules --model olmo2-1b-sft --split dev --max_new_tokens 512 \
      --batch_problems 2 --rl_oracle --rl_oracle_model $OUT/$tag --tag forkonly_$tag > $L/forkonly_$tag.log 2>&1
    r=$(grep -oE '"acc_lenient": [0-9.]+' $L/forkonly_$tag.log | tail -1); [ -n "$r" ] && { echo "  $r"; break; }
    echo "  failed: $(grep -oE 'OutOfMemoryError|Error[^:]*' $L/forkonly_$tag.log | tail -1)"
  done
done
git add results/screen/olmo2-1b-sft/dev/rl_rules_forkonly_* scripts/queue_forkonly_retry.sh 2>/dev/null
git commit -q -m "Fork-only inference with round-3 LoRAs on 1B SFT (dev)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q 2>&1 | tail -1
