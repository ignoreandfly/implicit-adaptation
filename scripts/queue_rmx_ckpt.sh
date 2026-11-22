#!/usr/bin/env bash
# ReasonMaxxer-style training WITH the paper's checkpoint selection: adapters every 20 optimizer steps, each merged and
# evaluated on dev; the best dev checkpoint (incl. step 0 = no training) is then evaluated on test500.
# Usage: MODEL=base|sft TAG=<name> bash scripts/queue_rmx_ckpt.sh <extra decision_lora args>
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HPY=$HOME/miniconda3/envs/ICCV/bin/python; OUT=/data/azfarm/siddhant_fork_lora; R=results/rmx_ckpt/$TAG; L=logs/queue
COT=" Let's think step by step."
if [ "$MODEL" = base ]; then M=olmo2-1b-base; GENS=results/screen/olmo2-1b-base/train/sample_prefix-let-s-think-step-by-step_T0.7_n8; PRE="$COT"
else M=olmo2-1b-sft; GENS=results/judge/olmo2-1b-sft_train1500_n8; PRE=""; fi
mkdir -p $R
pick_gpu () { while true; do g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
  awk -F', ' -v need="$1" '$2 >= need {print $1; exit}'); [ -n "$g" ] && { echo "$g"; return; }; sleep 60; done; }
acc () { grep -oE '"acc_lenient": [0-9.]+' $1 | tail -1 | grep -oE '[0-9.]+$'; }
evaluate () {  # model_path split out
  g=$(pick_gpu 6000)
  if [ "$MODEL" = base ]; then
    CUDA_VISIBLE_DEVICES=$g $HPY -m methods.run --method greedy --model $1 --prefix "$COT" --split $2 \
      --max_new_tokens 320 --batch_problems 1 --out $3 > $3.log 2>&1
  else
    CUDA_VISIBLE_DEVICES=$g $HPY -m methods.run --method rl_rules --rl_beta 0 --model $1 --format chat \
      --split $2 --max_new_tokens 512 --batch_problems 2 --out $3 > $3.log 2>&1
  fi
  acc $3.log
}
CK=$OUT/rmxck_$TAG
if [ ! -f $CK/decision_config.json ]; then
  g=$(pick_gpu 10000); echo "=== [gpu$g] train $TAG ($*) $(date)"
  CUDA_VISIBLE_DEVICES=$g $HPY -m train.decision_lora --gens $GENS --model $M --prefix "$PRE" --label gold \
    --save_every 20 "$@" --out $CK > $L/train_rmxck_$TAG.log 2>&1
fi
grep -E "rollouts;|gate |tau =" $L/train_rmxck_$TAG.log; cp $CK/decision_config.json $R/ 2>/dev/null || { echo TRAIN FAILED; tail -3 $L/train_rmxck_$TAG.log; exit 1; }
best=""; bestacc=-1
for ad in $(ls -d $CK/adapter_step* | sort -V); do
  s=$(basename $ad); mkdir -p $R/$s
  $HPY -m train.merge_adapter --model $M --adapter $ad --out $OUT/rmxck_tmp_$TAG > /dev/null 2>&1
  a=$(evaluate $OUT/rmxck_tmp_$TAG dev $R/$s/dev); echo "  $TAG $s dev $a"
  if awk "BEGIN{exit !($a > $bestacc)}"; then bestacc=$a; best=$ad; fi
done
echo "  best: $(basename $best) dev $bestacc"
$HPY -m train.merge_adapter --model $M --adapter $best --out $OUT/rmxck_tmp_$TAG > /dev/null 2>&1
echo "  $TAG $(basename $best) test500 $(evaluate $OUT/rmxck_tmp_$TAG test500 $R/best_test500)"
echo "=== done $TAG $(date)"
