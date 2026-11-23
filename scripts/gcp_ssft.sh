#!/usr/bin/env bash
# Self-SSFT on OLMo 2 1B SFT (train/ssft_lora.py; SSFT = Jia et al. ICLR 2026, arXiv 2510.05132).
# Question: do SSFT's global forking tokens + set matching, trained on the model's OWN correct samples,
# beat plain RFT on the same data (Pass@1 with <think1>; Cons@4 vs maj@4 at equal compute)?
#   SMOKE=1 bash scripts/gcp_ssft.sh     # ~15 min end-to-end check on 40 questions - run this first
#   bash scripts/gcp_ssft.sh             # full run (resumable; ~4-6 GPU-hours on one L4/A100)
#   STEPS="eval cons" bash scripts/gcp_ssft.sh
# Env: VPY = python of the vllm env, HPY = python of the hf/peft env (RUNBOOK section 1).
set -uo pipefail
cd "$(dirname "$0")/.."
VPY=${VPY:-python}; HPY=${HPY:-python}; MEM=${MEM:-0.85}
STEPS=${STEPS:-"data train eval cons"}
SMOKE=${SMOKE:-0}
if [ "$SMOKE" = 1 ]; then NQ=40; R=results/ssft_self_smoke; EVAL="--split dev --limit 40"; SPLIT=dev; LIM="--limit 40"
else NQ=1500; R=results/ssft_self; EVAL="--subset 500"; SPLIT=test; LIM=""; fi
CK=${CK:-checkpoints/ssft_self$([ "$SMOKE" = 1 ] && echo _smoke)}
NT=4
mkdir -p $R/eval $CK logs
SAMPLES=$R/sft_train${NQ}_n8
has () { [[ " $STEPS " == *" $1 "* ]]; }
acc () { grep -oE "'acc_lenient': [0-9.]+" $1 | tail -1; }
[ -f data/gsm8k/test.jsonl ] || { echo "prepare data first (RUNBOOK section 1)"; exit 1; }

if has data; then   # 8 SFT samples per training question (T 0.7), same recipe as the RFT-on-SFT runs
  [ -f $SAMPLES/train.generations.jsonl ] || $VPY -m evaluation.generate --backend vllm --model olmo2-1b-sft \
    --split train --subset $NQ --n_samples 8 --temperature 0.7 --gpu_memory_utilization $MEM --out $SAMPLES \
    > logs/ssft_data.log 2>&1
  echo "data: $(acc logs/ssft_data.log)"
fi

train () {  # tag, trainer module, args...
  local tag=$1 mod=$2; shift 2
  [ -f $CK/$tag/model.safetensors ] || $HPY -m $mod "$@" --out $CK/$tag > logs/ssft_train_$tag.log 2>&1
  [ -f $CK/$tag/model.safetensors ] && echo "trained $tag: $(grep -m1 'questions' logs/ssft_train_$tag.log)" \
    || echo "TRAINING FAILED $tag (logs/ssft_train_$tag.log)"
}
if has train; then
  for lab in gold majority; do
    train ssft_$lab       train.ssft_lora --gens $SAMPLES --model olmo2-1b-sft --label $lab --n_tags $NT --m $NT --assign match
    train rft_${lab}_m4   train.ssft_lora --gens $SAMPLES --model olmo2-1b-sft --label $lab --n_tags $NT --m $NT --assign none
  done
  train ssft_gold_random  train.ssft_lora --gens $SAMPLES --model olmo2-1b-sft --label gold --n_tags $NT --m $NT --assign random
  train rft_gold_all      train.rft_lora  --gens $SAMPLES --model olmo2-1b-sft --label gold --lr 1e-4 --epochs 2
fi

gen () {  # out-tag, model, extra args
  local t=$1 m=$2; shift 2
  [ -f $R/eval/$t/$SPLIT.metrics.json ] || $VPY -m evaluation.generate --backend vllm --model $m --format chat $EVAL \
    --gpu_memory_utilization $MEM "$@" --out $R/eval/$t > logs/ssft_eval_$t.log 2>&1
  echo "eval $t: $(acc logs/ssft_eval_$t.log)"
}
if has eval; then
  # untagged models: greedy (pass@1) + 4 samples at T 0.7 (maj@4 = same compute as one greedy path per tag)
  for m in olmo2-1b-sft olmo2-1b-instruct $CK/rft_gold_m4 $CK/rft_majority_m4 $CK/rft_gold_all; do
    t=$(basename $m)
    gen $t $m
    gen ${t}_n4 $m --n_samples 4 --temperature 0.7
  done
  # tagged models: greedy once per tag, and 1 sample at T 0.7 per tag
  for t in ssft_gold ssft_majority ssft_gold_random; do
    for i in $(seq 1 $NT); do
      gen ${t}_t$i $CK/$t --suffix "<think$i>"
      gen ${t}_t${i}_T0.7 $CK/$t --suffix "<think$i>" --temperature 0.7
    done
  done
fi

if has cons; then
  E=$R/eval
  tags () { local t=$1 sfx=${2:-}; echo "tags:$t$sfx=$(seq -s, -f "$E/${t}_t%g$sfx" 1 $NT | sed 's/,$//')"; }
  for lab in gold majority; do
    $HPY -m analysis.ssft_cons --split $SPLIT $(tags ssft_$lab) $(tags ssft_$lab _T0.7) \
      $([ $lab = gold ] && tags ssft_gold_random) samples:rft_${lab}_m4=$E/rft_${lab}_m4_n4 \
      $([ $lab = gold ] && echo samples:rft_gold_all=$E/rft_gold_all_n4) samples:sft=$E/olmo2-1b-sft_n4 \
      samples:instruct=$E/olmo2-1b-instruct_n4 --out $R/cons_$lab.json
  done
fi
echo "=== gcp_ssft done $(date)"
