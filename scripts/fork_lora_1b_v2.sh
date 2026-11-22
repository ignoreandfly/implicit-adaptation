#!/usr/bin/env bash
# Fork-LoRA round 2 (dev only first): bounded token-level DPO loss + low learning rates, after round 1's
# unbounded loss destroyed the model even with shuffled labels. Waits for round 1 to finish.
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VPY=$HOME/miniconda3/envs/implicit-vllm/bin/python; HPY=$HOME/miniconda3/envs/ICCV/bin/python
PAIRS=data/fork_pairs/olmo2-1b-sft_train1000.jsonl; OUT=/data/azfarm/siddhant_fork_lora; R=results/fork_lora
run () {  # tag, args
  tag=$1; shift
  echo "=== train $tag $(date)"
  $HPY -m train.fork_lora --pairs $PAIRS --model olmo2-1b-sft "$@" --out $OUT/$tag 2>&1 | grep -E "training forks|saved|Error" | tail -2
  $VPY -m evaluation.generate --backend vllm --model $OUT/$tag --format chat --split dev --gpu_memory_utilization 0.5 \
    --out $R/$tag/greedy_dev 2>&1 | grep -oE "'acc_lenient': [0-9.]+" | sed "s/^/  $tag dev greedy /"
  cp $OUT/$tag/fork_lora_config.json $R/$tag/ 2>/dev/null
}
run dpo_major_lr2e-5_b0.5 --loss dpo --label majority --lr 2e-5 --beta 0.5 --epochs 1
run dpo_major_lr5e-6_b0.5 --loss dpo --label majority --lr 5e-6 --beta 0.5 --epochs 1
run dpo_major_lr2e-5_b0.1 --loss dpo --label majority --lr 2e-5 --beta 0.1 --epochs 1
run adv_major_lr1e-5      --loss adv --label majority --lr 1e-5 --epochs 1
run dpo_shuffled_lr2e-5_b0.5 --loss dpo --label majority --shuffle --lr 2e-5 --beta 0.5 --epochs 1
echo "=== done $(date)  (SFT baseline dev greedy = 0.695)"
