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
SIZE=${SIZE:-1b}                       # 1b or 7b
SFTM=olmo2-${SIZE}-sft; INSTM=olmo2-${SIZE}-instruct
SUF=$([ "$SIZE" = 1b ] || echo "_$SIZE")
if [ "$SMOKE" = 1 ]; then NQ=40; R=results/ssft_self${SUF}_smoke; EVAL="--split dev --limit 40"; SPLIT=dev; LIM="--limit 40"
else NQ=1500; R=results/ssft_self${SUF}; EVAL="--subset 500"; SPLIT=test; LIM=""; fi
CK=${CK:-checkpoints/ssft_self${SUF}$([ "$SMOKE" = 1 ] && echo _smoke)}
NT=4
mkdir -p $R/eval $CK logs
SAMPLES=$R/sft_train${NQ}_n8
has () { [[ " $STEPS " == *" $1 "* ]]; }
acc () { grep -oE "'acc_lenient': [0-9.]+" $1 | tail -1; }
[ -f data/gsm8k/test.jsonl ] || { echo "prepare data first (RUNBOOK section 1)"; exit 1; }

if has data; then   # 8 SFT samples per training question (T 0.7), same recipe as the RFT-on-SFT runs
  [ -f $SAMPLES/train.generations.jsonl ] || $VPY -m evaluation.generate --backend vllm --model $SFTM \
    --split train --subset $NQ --n_samples 8 --temperature 0.7 --gpu_memory_utilization $MEM --out $SAMPLES \
    > logs/ssft${SUF}_data.log 2>&1
  echo "data: $(acc logs/ssft${SUF}_data.log)"
fi

gen () {  # out-tag, model, extra args
  local t=$1 m=$2; shift 2
  [ -f $R/eval/$t/$SPLIT.metrics.json ] || $VPY -m evaluation.generate --backend vllm --model $m --format chat $EVAL \
    --gpu_memory_utilization $MEM "$@" --out $R/eval/$t > logs/ssft${SUF}_eval_$t.log 2>&1
  echo "eval $t: $(acc logs/ssft${SUF}_eval_$t.log)"
}
evals () {  # out-tag, model, tagged(0/1): greedy + T0.7 paths; untagged = greedy + 4 samples (= one path per tag)
  local t=$1 m=$2
  if [ "$3" = 1 ]; then
    for i in $(seq 1 $NT); do
      gen ${t}_t$i $m --suffix "<think$i>"
      gen ${t}_t${i}_T0.7 $m --suffix "<think$i>" --temperature 0.7
    done
  else
    gen $t $m; gen ${t}_n4 $m --n_samples 4 --temperature 0.7
  fi
}
done_evals () { [ -f $R/eval/$1$([ "$2" = 1 ] && echo _t${NT}_T0.7 || echo _n4)/$SPLIT.metrics.json ]; }
# train -> evaluate -> delete the checkpoint (the VM disk holds ~1 model at a time); configs kept in $R/configs
model () {  # tag, tagged(0/1), trainer module, args...
  local tag=$1 tg=$2 mod=$3; shift 3
  done_evals $tag $tg && { echo "done $tag"; return; }
  if has train && ! ls $CK/$tag/*_config.json >/dev/null 2>&1; then
    rm -rf $CK/$tag   # partial save from an interrupted run
    $HPY -m $mod "$@" --out $CK/$tag > logs/ssft${SUF}_train_$tag.log 2>&1
  fi
  ls $CK/$tag/*_config.json >/dev/null 2>&1 || { echo "TRAINING FAILED $tag (logs/ssft${SUF}_train_$tag.log)"; return; }
  echo "trained $tag: $(grep -m1 'questions\|training solutions' logs/ssft${SUF}_train_$tag.log)"
  mkdir -p $R/configs; cp $CK/$tag/ssft_config.json $R/configs/$tag.json 2>/dev/null || cp $CK/$tag/rft_config.json $R/configs/$tag.json 2>/dev/null
  has eval && evals $tag $CK/$tag $tg
  done_evals $tag $tg && [ "${KEEP:-0}" != 1 ] && rm -rf $CK/$tag
}
if has eval; then evals $SFTM $SFTM 0; evals $INSTM $INSTM 0; fi
for lab in gold majority; do
  model ssft_$lab     1 train.ssft_lora --gens $SAMPLES --model $SFTM --label $lab --n_tags $NT --m $NT --assign match
  model rft_${lab}_m4 0 train.ssft_lora --gens $SAMPLES --model $SFTM --label $lab --n_tags $NT --m $NT --assign none
done
model ssft_gold_random 1 train.ssft_lora --gens $SAMPLES --model $SFTM --label gold --n_tags $NT --m $NT --assign random
model rft_gold_all     0 train.rft_lora  --gens $SAMPLES --model $SFTM --label gold --lr 1e-4 --epochs 2

if has cons; then
  E=$R/eval
  tags () { local t=$1 sfx=${2:-}; echo "tags:$t$sfx=$(seq -s, -f "$E/${t}_t%g$sfx" 1 $NT | sed 's/,$//')"; }
  for lab in gold majority; do
    $HPY -m analysis.ssft_cons --split $SPLIT $(tags ssft_$lab) $(tags ssft_$lab _T0.7) \
      $([ $lab = gold ] && tags ssft_gold_random) samples:rft_${lab}_m4=$E/rft_${lab}_m4_n4 \
      $([ $lab = gold ] && echo samples:rft_gold_all=$E/rft_gold_all_n4) samples:sft=$E/${SFTM}_n4 \
      samples:instruct=$E/${INSTM}_n4 --out $R/cons_$lab.json
  done
fi
echo "=== gcp_ssft done $(date)"
# stop the VM when everything is done (set SHUTDOWN=1 on a dedicated cloud VM so it stops billing)
[ "${SHUTDOWN:-0}" = 1 ] && [ "$SMOKE" != 1 ] && sudo shutdown -h +2 "gcp_ssft finished"
