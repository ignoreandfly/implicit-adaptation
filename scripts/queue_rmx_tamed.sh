#!/usr/bin/env bash
# Fair test of ReasonMaxxer-style decision-point training on OLMo 2 1B Base + CoT (see README / memory: experiment
# discipline). Differences from the earlier quick runs:
#   data       = their recipe: 300 training problems x 20 samples at T 0.6 (was 8 samples at T 0.7)
#   sparsity   = 5% of response tokens for both gates (entropy vs our fork top-2 margin), + their tau 1.4 for reference
#   selection  = adapters every 50 steps, picked on heldout500 (500 held-out TRAIN questions, idx 6500-6999), not dev
#   decision   = best checkpoint on the FULL test (1,319); paired test vs prompt-only and RFT afterwards
# Tamed ReasonMaxxer on the same 50-problem recipe data: negatives scaled by beta_neg 0.3, lr 5e-5, adapters every 25 steps (all variants peaked at the first checkpoint before), 200-problem variant, soft fork weighting under the same fixes; then sampled pass@1 (T 0.6, their metric) on test500 for the references and the best runs.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HPY=$HOME/miniconda3/envs/ICCV/bin/python; OUT=/data/azfarm/siddhant_fork_lora; R=results/rmx_fair; L=logs/queue
COT=" Let's think step by step."; M=olmo2-1b-base; GPU=${GPU:?set GPU}
GENS=results/screen/olmo2-1b-base/train/sample_prefix-let-s-think-step-by-step_T0.6_n20
mkdir -p $R
wait_mem () { while true; do f=$(nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ');
  [ "$f" -ge "$1" ] && return; sleep 60; done; }
acc () { grep -oE '"acc_lenient": [0-9.]+' $1 | tail -1 | grep -oE '[0-9.]+$'; }
evaluate () {  # model split outdir  (retries on OOM)
  for t in 1 2 3; do
    wait_mem 6000
    CUDA_VISIBLE_DEVICES=$GPU $HPY -m methods.run --method greedy --model $1 --prefix "$COT" --split $2 \
      --max_new_tokens 320 --batch_problems 1 --out $3 > $3.log 2>&1
    a=$(acc $3.log); [ -n "$a" ] && { echo $a; return; }; sleep 60
  done; echo FAIL; }

# 1. data
if [ ! -f $GENS/train.generations.jsonl ]; then
  wait_mem 8000; echo "=== [gpu$GPU] sample 300 x 20 at T 0.6 $(date)"
  CUDA_VISIBLE_DEVICES=$GPU $HPY -m methods.run --method sample --model $M --split train --limit 300 --n 20 \
    --temperature 0.6 --prefix "$COT" --max_new_tokens 320 --batch_problems 4 > $L/rmx_fair_sample.log 2>&1
fi
[ -f $GENS/train.generations.jsonl ] || { echo "sampling failed"; tail -3 $L/rmx_fair_sample.log; exit 1; }
echo "data: $(grep -oE '"acc_lenient": [0-9.]+' $GENS/train.metrics.json | head -1) (mean sample accuracy)"

# 2. prompt-only reference on the selection split
[ -f $R/prompt_only_heldout.log ] || echo "prompt-only heldout500: $(evaluate $M heldout500 $R/prompt_only_heldout)"

# 3. training runs with checkpoint selection on heldout500, then full test
run () {  # tag, decision_lora args
  tag=$1; shift; CK=$OUT/rmxfair_$tag; mkdir -p $R/$tag
  if [ ! -f $CK/decision_config.json ]; then
    wait_mem 12000; echo "=== [gpu$GPU] train $tag ($*) $(date)"
    CUDA_VISIBLE_DEVICES=$GPU $HPY -m train.decision_lora --gens $GENS --model $M --prefix "$COT" --label gold \
      --save_every 50 "$@" --out $CK > $L/train_rmxfair_$tag.log 2>&1
  fi
  grep -E "rollouts;|gate " $L/train_rmxfair_$tag.log
  cp $CK/decision_config.json $R/$tag/ 2>/dev/null || { echo "  TRAIN FAILED"; tail -3 $L/train_rmxfair_$tag.log; return; }
  best=""; bestacc=-1
  for ad in $(ls -d $CK/adapter_step* | sort -V); do
    s=$(basename $ad)
    $HPY -m train.merge_adapter --model $M --adapter $ad --out $OUT/rmxfair_tmp_$tag > /dev/null 2>&1
    a=$(evaluate $OUT/rmxfair_tmp_$tag heldout500 $R/$tag/${s}_heldout); echo "  $tag $s heldout500 $a"
    [ "$a" != FAIL ] && awk "BEGIN{exit !($a > $bestacc)}" && { bestacc=$a; best=$ad; }
  done
  echo "  $tag best $(basename $best) heldout500 $bestacc"
  $HPY -m train.merge_adapter --model $M --adapter $best --out $OUT/rmxfair_tmp_$tag > /dev/null 2>&1
  echo "  $tag $(basename $best) FULL TEST $(evaluate $OUT/rmxfair_tmp_$tag test $R/$tag/best_test)"
}
T="--beta_neg 0.3 --lr 5e-5 --save_every 25"
for r in ${RUNS:-ent05_tamed ent05_tamed_np200 wg2_tamed}; do
  case $r in
    ent05_tamed)       run ent05_tamed       --gate entropy --share 0.05 $T ;;
    ent05_tamed_np200) run ent05_tamed_np200 --gate entropy --share 0.05 --n_problems 200 $T ;;
    wg2_tamed)         run wg2_tamed         --gate weight --gamma 2 $T ;;
  esac
done
# sampled pass@1 at T 0.6 (ReasonMaxxer's metric), 4 samples per question, test500
sampled () {  # name model_dir
  [ -f $R/sampled/$1/test500.metrics.json ] && { echo "  sampled $1 $(grep -oE '"acc_single_sample": [0-9.]+' $R/sampled/$1/test500.metrics.json)"; return; }
  wait_mem 6000; mkdir -p $R/sampled
  CUDA_VISIBLE_DEVICES=$GPU $HPY -m methods.run --method sample --model $2 --prefix "$COT" --split test500 --n 4 \
    --temperature 0.6 --max_new_tokens 320 --batch_problems 4 --out $R/sampled/$1 > $R/sampled/$1.log 2>&1
  echo "  sampled pass@1 (T 0.6) $1: $(grep -oE '"acc_single_sample": [0-9.]+' $R/sampled/$1/test500.metrics.json)"
}
best_merged () {  # tag -> merged dir of its best adapter
  b=$(grep -h "$1 best" $L/../queue/rmx_tamed*.log $L/rmx_fair.log 2>/dev/null | tail -1 | awk '{print $3}')
  [ -z "$b" ] && b=adapter_step50
  $HPY -m train.merge_adapter --model $M --adapter $OUT/rmxfair_$1/$b --out $OUT/rmx_sampled_tmp > /dev/null 2>&1
  echo $OUT/rmx_sampled_tmp; }
sampled prompt_only $M
sampled rft686 $OUT/base_rft_gold
for t in entropy_share05 ent05_tamed wg2_tamed; do sampled $t $(best_merged $t); done
echo "=== done $(date)"
