#!/usr/bin/env bash
# Early local signal for faithful ReasonMaxxer (train/decision_lora.py, reference-code recipe) on EXISTING samples
# (n = 8 per problem, T 0.7, instead of their 20 at T 0.6). Compare against RFT on the same samples:
#   Base + CoT: RFT gold 45.6 test500 (prompt-only 37.0);  SFT: RFT gold 63.4 test500 (SFT 59.8, same decoder).
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HPY=$HOME/miniconda3/envs/ICCV/bin/python; OUT=/data/azfarm/siddhant_fork_lora; R=results/rmx_local; L=logs/queue
COT=" Let's think step by step."
BASE_GENS=results/screen/olmo2-1b-base/train/sample_prefix-let-s-think-step-by-step_T0.7_n8
SFT_GENS=results/judge/olmo2-1b-sft_train1500_n8
mkdir -p $R
pick_gpu () { while true; do g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
  awk -F', ' -v need="$1" '$2 >= need {print $1; exit}'); [ -n "$g" ] && { echo "$g"; return; }; sleep 90; done; }
acc () { grep -oE '"acc_lenient": [0-9.]+' $1 | tail -1; }
job () {  # tag model gens prefix extra-args
  tag=$1; model=$2; gens=$3; pre=$4; shift 4
  g=$(pick_gpu 14000); echo "=== [gpu$g] train $tag $(date)"
  CUDA_VISIBLE_DEVICES=$g $HPY -m train.decision_lora --gens $gens --model $model --prefix "$pre" --label gold "$@" \
    --out $OUT/$tag > $L/train_$tag.log 2>&1
  grep -E "rollouts;|tau =" $L/train_$tag.log
  mkdir -p $R/$tag; cp $OUT/$tag/decision_config.json $R/$tag/ 2>/dev/null || { echo "  TRAIN FAILED"; tail -3 $L/train_$tag.log; return; }
  for split in dev test500; do
    g=$(pick_gpu 6000)
    if [ "$model" = olmo2-1b-base ]; then
      CUDA_VISIBLE_DEVICES=$g $HPY -m methods.run --method greedy --model $OUT/$tag --prefix "$COT" --split $split \
        --max_new_tokens 320 --batch_problems 1 --out $R/$tag/merged_$split > $L/eval_${tag}_$split.log 2>&1
    else
      CUDA_VISIBLE_DEVICES=$g $HPY -m methods.run --method rl_rules --rl_beta 0 --model $OUT/$tag --format chat \
        --split $split --max_new_tokens 512 --batch_problems 2 --out $R/$tag/merged_$split > $L/eval_${tag}_$split.log 2>&1
    fi
    echo "  $tag $split $(acc $L/eval_${tag}_$split.log)"
  done
}
job rmx_base_gold    olmo2-1b-base $BASE_GENS "$COT"
job rmx_sft_gold     olmo2-1b-sft  $SFT_GENS  ""
job rmx_base_posonly olmo2-1b-base $BASE_GENS "$COT" --positive_only
echo "=== done $(date)"
