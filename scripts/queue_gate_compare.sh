#!/usr/bin/env bash
# Decision-point criterion at matched sparsity: ReasonMaxxer's entropy gate vs our fork (top-2 margin) gate, both
# gating 5% of response tokens, same data / recipe as scripts/queue_rmx_local.sh. Usage: MODEL=base|sft bash ...
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HPY=$HOME/miniconda3/envs/ICCV/bin/python; OUT=/data/azfarm/siddhant_fork_lora; R=results/rmx_local; L=logs/queue
COT=" Let's think step by step."
if [ "$MODEL" = base ]; then M=olmo2-1b-base; GENS=results/screen/olmo2-1b-base/train/sample_prefix-let-s-think-step-by-step_T0.7_n8; PRE="$COT"
else M=olmo2-1b-sft; GENS=results/judge/olmo2-1b-sft_train1500_n8; PRE=""; fi
pick_gpu () { while true; do g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
  awk -F', ' -v need="$1" '$2 >= need {print $1; exit}'); [ -n "$g" ] && { echo "$g"; return; }; sleep 60; done; }
acc () { grep -oE '"acc_lenient": [0-9.]+' $1 | tail -1; }
for gate in margin entropy; do
  tag=rmx_${MODEL}_${gate}_share05
  g=$(pick_gpu 10000); echo "=== [gpu$g] train $tag $(date)"
  CUDA_VISIBLE_DEVICES=$g $HPY -m train.decision_lora --gens $GENS --model $M --prefix "$PRE" --label gold \
    --gate $gate --share 0.05 --out $OUT/$tag > $L/train_$tag.log 2>&1
  grep -E "rollouts;|gate " $L/train_$tag.log
  mkdir -p $R/$tag; cp $OUT/$tag/decision_config.json $R/$tag/ 2>/dev/null || { echo "  TRAIN FAILED"; tail -3 $L/train_$tag.log; continue; }
  for split in dev test500; do
    g=$(pick_gpu 6000)
    if [ "$MODEL" = base ]; then
      CUDA_VISIBLE_DEVICES=$g $HPY -m methods.run --method greedy --model $OUT/$tag --prefix "$COT" --split $split \
        --max_new_tokens 320 --batch_problems 1 --out $R/$tag/merged_$split > $L/eval_${tag}_$split.log 2>&1
    else
      CUDA_VISIBLE_DEVICES=$g $HPY -m methods.run --method rl_rules --rl_beta 0 --model $OUT/$tag --format chat \
        --split $split --max_new_tokens 512 --batch_problems 2 --out $R/$tag/merged_$split > $L/eval_${tag}_$split.log 2>&1
    fi
    echo "  $tag $split $(acc $L/eval_${tag}_$split.log)"
  done
done
echo "=== done $(date)"
