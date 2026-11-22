#!/usr/bin/env bash
# "Where should the self-training signal go?" on OLMo 2 1B Base + zero-shot CoT, anchored on RFT (HANDOFF_RMX.md).
# Same data (1,000 train questions x 8 samples, T 0.7), same optimizer (LoRA r32 q/k/v/o, lr 1e-4, 2 epochs, accum 8),
# correct solutions only with weight 1 (positive-only: cannot drift the way negatives can). Only the TOKENS that
# receive the loss differ; every other position gets the KL anchor to the frozen model:
#   rft_ref     : train/rft_lora.py, plain RFT (the anchor; 45.6 test500 / 43.7 full test on the main machine)
#   all         : decision_lora --gate all            (RFT through the same trainer: sanity check vs rft_ref)
#   margin05    : decision_lora --gate margin --share 0.05   (our fork criterion: smallest top-2 margin)
#   entropy05   : decision_lora --gate entropy --share 0.05  (ReasonMaxxer's criterion, same sparsity)
#   margin20    : decision_lora --gate margin --share 0.20   (sparsity check)
# Checkpoints every 150 optimizer steps (+ final) are picked on heldout500 (train questions 6500-6999, never trained
# on); the pick is evaluated once on the FULL test (1,319). Resumable: rerun the script and finished steps are skipped.
#   GPU=0 bash scripts/rmx_gate_rft.sh               # all runs
#   GPU=0 RUNS="margin05 entropy05" bash scripts/rmx_gate_rft.sh
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HPY=${HPY:-python}; GPU=${GPU:?set GPU=<one free gpu index>}
OUT=${OUT:-checkpoints/rmx_gate_rft}; R=results/rmx_gate_rft; L=logs/rmx_gate_rft
RUNS=${RUNS:-"rft_ref all margin05 entropy05 margin20"}
COT=" Let's think step by step."; M=olmo2-1b-base
GENS=results/screen/olmo2-1b-base/train/sample_prefix-let-s-think-step-by-step_T0.7_n8
mkdir -p $R $L $OUT
wait_mem () { while true; do f=$(nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ');
  [ "$f" -ge "$1" ] && return; sleep 60; done; }
acc () { grep -oE '"acc_lenient": [0-9.]+' $1 | tail -1 | grep -oE '[0-9.]+$'; }
evaluate () {  # model split out -> prints accuracy (retries: shared GPUs)
  if [ -f $3/$2.metrics.json ]; then acc $3/$2.metrics.json; return; fi
  for t in 1 2 3; do
    wait_mem 6000
    CUDA_VISIBLE_DEVICES=$GPU $HPY -m methods.run --method greedy --model $1 --prefix "$COT" --split $2 \
      --max_new_tokens 320 --batch_problems 1 --out $3 > $3.log 2>&1
    a=$(acc $3.log); [ -n "$a" ] && { echo $a; return; }; sleep 60
  done; echo FAIL; }

# 0. selection split (deterministic from data/gsm8k/train.jsonl, RUNBOOK section 1)
[ -f data/gsm8k/heldout500.jsonl ] || $HPY -c "
import json; rows=[l for l in open('data/gsm8k/train.jsonl')]; open('data/gsm8k/heldout500.jsonl','w').writelines(rows[6500:7000])"
[ $(wc -l < data/gsm8k/heldout500.jsonl) = 500 ] || { echo "bad heldout500"; exit 1; }

# 1. training samples (same command as scripts/queue_rft_baseline.sh)
if [ ! -f $GENS/train.generations.jsonl ]; then
  wait_mem 8000; echo "=== sample Base+CoT 1,000 x 8 at T 0.7 $(date)"
  CUDA_VISIBLE_DEVICES=$GPU $HPY -m methods.run --method sample --model $M --split train --limit 1000 --n 8 \
    --temperature 0.7 --prefix "$COT" --max_new_tokens 320 --batch_problems 8 > $L/sample.log 2>&1
fi
[ -f $GENS/train.generations.jsonl ] || { echo "sampling failed: $L/sample.log"; exit 1; }

# 2. references on this machine's pipeline
echo "prompt-only heldout500 $(evaluate $M heldout500 $R/prompt_only_heldout500)"
echo "prompt-only FULL TEST  $(evaluate $M test $R/prompt_only_test)"

# 3. runs
pick_eval () {  # tag ckpt_dir: evaluate adapters on heldout500, best one on the full test
  tag=$1; CK=$2; best=""; bestacc=-1
  for ad in $(ls -d $CK/adapter_step* | sort -V); do
    s=$(basename $ad)
    if [ ! -f $R/$tag/${s}_heldout500/heldout500.metrics.json ]; then
      $HPY -m train.merge_adapter --model $M --adapter $ad --out $OUT/tmp_merged > /dev/null 2>&1
    fi
    a=$(evaluate $OUT/tmp_merged heldout500 $R/$tag/${s}_heldout500); echo "  $tag $s heldout500 $a"
    [ "$a" != FAIL ] && awk "BEGIN{exit !($a > $bestacc)}" && { bestacc=$a; best=$ad; }
  done
  echo "  $tag best $(basename $best) heldout500 $bestacc"
  echo "$(basename $best)" > $R/$tag/best.txt
  if [ ! -f $R/$tag/best_test/test.metrics.json ]; then
    $HPY -m train.merge_adapter --model $M --adapter $best --out $OUT/tmp_merged > /dev/null 2>&1
  fi
  echo "  $tag $(basename $best) FULL TEST $(evaluate $OUT/tmp_merged test $R/$tag/best_test)"
}
for run in $RUNS; do
  mkdir -p $R/$run
  if [ $run = rft_ref ]; then
    [ -f $OUT/rft_ref/model.safetensors ] || { wait_mem 10000; echo "=== train rft_ref $(date)";
      CUDA_VISIBLE_DEVICES=$GPU $HPY -m train.rft_lora --gens $GENS --split train --model $M --prefix "$COT" \
        --label gold --lr 1e-4 --epochs 2 --out $OUT/rft_ref > $L/train_rft_ref.log 2>&1; }
    echo "  rft_ref (final, as in the main machine's RFT) heldout500 $(evaluate $OUT/rft_ref heldout500 $R/rft_ref/heldout500)"
    echo "  rft_ref FULL TEST $(evaluate $OUT/rft_ref test $R/rft_ref/best_test)"
    continue
  fi
  case $run in
    all)       G="--gate all" ;;
    margin05)  G="--gate margin --share 0.05" ;;
    entropy05) G="--gate entropy --share 0.05" ;;
    margin20)  G="--gate margin --share 0.20" ;;
    *) echo "unknown run $run"; continue ;;
  esac
  CK=$OUT/$run
  if [ ! -f $CK/decision_config.json ]; then
    wait_mem 12000; echo "=== train $run ($G) $(date)"
    CUDA_VISIBLE_DEVICES=$GPU $HPY -m train.decision_lora --gens $GENS --model $M --prefix "$COT" --label gold \
      --rft_weights --n_problems 0 --trim 1 --max_steps 0 --epochs 2 --warmup 50 --lr 1e-4 \
      --shuffle --save_every 150 $G --out $CK > $L/train_$run.log 2>&1
  fi
  grep -E "rollouts;|gate " $L/train_$run.log
  cp $CK/decision_config.json $R/$run/ 2>/dev/null || { echo "  TRAIN FAILED ($L/train_$run.log)"; tail -3 $L/train_$run.log; continue; }
  pick_eval $run $CK
done
echo "=== rmx_gate_rft done $(date)"
