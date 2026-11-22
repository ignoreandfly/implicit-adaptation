#!/usr/bin/env bash
# Fork-LoRA round 3 (dev): bounded token-level DPO loss at learning rates large enough to move bf16 logits
# (round 2 at 5e-6..2e-5 did not change the model at all), 2 epochs. Also re-runs the lr=0 sanity eval.
# vLLM memory share 0.3: GPU 0 is shared with other users' jobs.
set -uo pipefail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VPY=$HOME/miniconda3/envs/implicit-vllm/bin/python; HPY=$HOME/miniconda3/envs/ICCV/bin/python
PAIRS=data/fork_pairs/olmo2-1b-sft_train1000.jsonl; OUT=/data/azfarm/siddhant_fork_lora; R=results/fork_lora
ev () {  # path, tag
  $VPY -m evaluation.generate --backend vllm --model $1 --format chat --split dev --gpu_memory_utilization 0.3 \
    --out $R/$2/greedy_dev > logs/eval_$2.log 2>&1
  grep -oE "'acc_lenient': [0-9.]+" logs/eval_$2.log | tail -1 | sed "s/^/  $2 dev greedy /" || true
  grep -qE "acc_lenient" logs/eval_$2.log || echo "  $2 EVAL FAILED (see logs/eval_$2.log)"
}
echo "=== sanity lr=0 eval $(date)"; ev $OUT/sanity_lr0 sanity_lr0
run () {
  tag=$1; shift
  echo "=== train $tag $(date)"
  $HPY -m train.fork_lora --pairs $PAIRS --model olmo2-1b-sft "$@" --out $OUT/$tag > logs/train_$tag.log 2>&1
  grep -E "training forks|saved" logs/train_$tag.log | tail -2
  grep -E "'step'" logs/train_$tag.log | tail -2
  ev $OUT/$tag $tag
  mkdir -p $R/$tag; cp $OUT/$tag/fork_lora_config.json $R/$tag/ 2>/dev/null
}
run dpo_major_lr1e-4_e2   --loss dpo --label majority --lr 1e-4 --beta 0.5 --epochs 2
run dpo_major_lr3e-4_e2   --loss dpo --label majority --lr 3e-4 --beta 0.5 --epochs 2
run dpo_gold_lr1e-4_e2    --loss dpo --label gold --lr 1e-4 --beta 0.5 --epochs 2
run dpo_shuffled_lr1e-4_e2 --loss dpo --label majority --shuffle --lr 1e-4 --beta 0.5 --epochs 2
echo "=== done $(date)  (SFT baseline dev greedy = 0.695)"
