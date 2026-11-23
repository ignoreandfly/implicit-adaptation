#!/usr/bin/env bash
# Seeds (LoRA init; the data and curriculum order are fixed by the recipe) + 2x2 of token credit, OLMo 2 1B Base + CoT,
# 50-problem ReasonMaxxer data. Everything built and evaluated in fp32 with one decoder (methods.run greedy, batch 4).
#   rft50  : all tokens of correct solutions, lr 1e-4        tamed : entropy gate 5%, beta_neg 0.3, lr 5e-5
#   P{h,u}N{l,h}: EAPO weights (kappa $KAPPA), beta_neg 0.3, lr 5e-5; positives high-entropy / uniform, negatives low / high
#   PhNl = EAPO's rule, PhNh = symmetric entropy credit (soft analogue of ReasonMaxxer)
# Checkpoints 50/100/last picked on heldout500; the pick is evaluated on fresh1000 (GSM8K train 4000-4999, unused so far)
# and SVAMP (300). The GSM8K test set is NOT used. Any failure prints FAILED.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HPY=$HOME/miniconda3/envs/ICCV/bin/python; D=${D:-/data/azfarm/siddhant_fork_lora/seeds2x2}; R=${R:-results/seeds_2x2}; L=logs/queue; M=olmo2-1b-base
COT=" Let's think step by step."; GENS=results/screen/olmo2-1b-base/train/sample_prefix-let-s-think-step-by-step_T0.6_n20
SMOKE=${SMOKE:-0}; LIM=""; X=""; [ $SMOKE = 1 ] && { LIM="--limit 8"; X="--max_steps 2 --limit 30"; }
KAPPA=${KAPPA:-1.3862944}; SEEDS=${SEEDS:-"0 1 2"}; ARMS=${ARMS:-"rft50 tamed PhNl PhNh PuNl PuNh"}
mkdir -p $D $R
pick_gpu () { while true; do g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' -v n=$1 '$2 >= n {print $2, $1}' | sort -rn | awk 'NR==1{print $2}'); [ -n "$g" ] && { echo $g; return; }; sleep 60; done; }
acc () { $HPY -c "import json,sys; print(json.load(open(sys.argv[1]))['acc_lenient'])" $1; }
ev () {  # out split adapter|none
  local o=$1 sp=$2 ad=$3
  for t in 1 2 3 4 5 6 7 8; do
    [ -f $o/$sp.metrics.json ] && break
    if [ $ad = none ]; then T=""; else T="--terms $ad:1"; fi
    rm -rf $D/tmp; $HPY -m analysis.lora_combo --model $M $T --out $D/tmp > $L/seeds_build.log 2>&1 || { echo "FAILED build $o" >&2; tail -1 $L/seeds_build.log >&2; echo FAIL; return; }
    g=$(pick_gpu 9000)
    IA_DTYPE=float32 CUDA_VISIBLE_DEVICES=$g $HPY -m methods.run --method greedy --model $D/tmp --format plain --prefix "$COT" --split $sp $LIM --max_new_tokens 320 --batch_problems 4 --out $o > $L/seeds_eval.log 2>&1 || { tail -1 $L/seeds_eval.log | cut -c1-140 >&2; sleep 120; }
  done
  [ -f $o/$sp.metrics.json ] && acc $o/$sp.metrics.json || { echo "FAILED eval $o $sp" >&2; echo FAIL; }; }
for sp in heldout500 fresh1000 svamp; do echo "prompt_only $sp $(ev $R/prompt_only/$sp $sp none)"; done
E="--gate eapo --kappa $KAPPA --beta_neg 0.3 --lr 5e-5"
for s in $SEEDS; do for a in $ARMS; do
  case $a in
    rft50) A="--gate all --positive_only --lr 1e-4" ;;  tamed) A="--gate entropy --share 0.05 --beta_neg 0.3 --lr 5e-5" ;;
    PhNl) A="$E --s_pos 1 --s_neg -1" ;; PhNh) A="$E --s_pos 1 --s_neg 1" ;; PuNl) A="$E --s_pos 0 --s_neg -1" ;; PuNh) A="$E --s_pos 0 --s_neg 1" ;;
  esac
  n=${a}_s$s; C=$D/$n
  for t in 1 2 3 4 5 6; do
    [ -f $C/decision_config.json ] && break
    g=$(pick_gpu 11000); echo "=== [gpu$g] train $n $(date +%T)"
    CUDA_VISIBLE_DEVICES=$g $HPY -m train.decision_lora --gens $GENS --model $M --prefix "$COT" --label gold --seed $s --save_every 50 --adapter_only $A $X --out $C > $L/seeds_train_$n.log 2>&1 || { tail -1 $L/seeds_train_$n.log | cut -c1-140; sleep 120; }
  done
  [ -f $C/decision_config.json ] || { echo "FAILED train $n"; continue; }
  best=""; ba=-1
  for ad in $(ls -d $C/adapter_step* | sort -V); do
    v=$(ev $R/$n/$(basename $ad)/heldout500 heldout500 $ad); echo "  $n $(basename $ad) heldout500 $v"
    [ "$v" != FAIL ] && awk "BEGIN{exit !($v > $ba)}" && { ba=$v; best=$ad; }
  done
  [ -z "$best" ] && { echo "FAILED $n"; continue; }
  echo "  $n PICK $(basename $best) heldout $ba | fresh1000 $(ev $R/$n/best/fresh1000 fresh1000 $best) | svamp $(ev $R/$n/best/svamp svamp $best) $(date +%T)"
done; done
rm -rf $D/tmp
$HPY - $R <<'PY'
import json, glob, sys, collections, statistics as st
R = sys.argv[1]; v = collections.defaultdict(lambda: collections.defaultdict(list))
for f in sorted(glob.glob(f"{R}/*_s*/best/*/*.metrics.json")):
    a = f[len(R)+1:].split("/")[0].rsplit("_s", 1)[0]; v[a][f.split("/")[-2]].append(json.load(open(f))["acc_lenient"])
for a, d in v.items(): print(a, {k: f"mean {st.mean(x):.4f} sd {st.pstdev(x) if len(x) < 2 else st.stdev(x):.4f} n={len(x)} {[round(i,3) for i in x]}" for k, x in d.items()})
PY
echo "=== done $(date)"
