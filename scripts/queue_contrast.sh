#!/usr/bin/env bash
# Contrastive weight extrapolation from OLMo 2 1B SFT, all built and evaluated in fp32, batch 16, one decoder.
# Grid chosen on heldout500 (train 6500-6999); only the pick of each family goes to the full test. Any failed step prints FAILED.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True IA_DTYPE=float32
HPY=$HOME/miniconda3/envs/ICCV/bin/python; D=${D:-/data/azfarm/siddhant_fork_lora/contrast}; R=${R:-results/contrast}; L=logs/queue
GENS=results/judge/olmo2-1b-sft_train1500_n8; SMOKE=${SMOKE:-0}; X=""; LIM=""; [ $SMOKE = 1 ] && { X="--limit_ex 16"; LIM="--limit 16"; }
mkdir -p $D $R
pick_gpu () { while true; do g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' -v n=$1 '$2 >= n {print $2, $1}' | sort -rn | awk 'NR==1{print $2}'); [ -n "$g" ] && { echo $g; return; }; sleep 60; done; }
[ -f $R/data/wrong/train.generations.jsonl ] || $HPY -m train.contrast_data --gens $GENS --out $R/data
for s in gold_all correct wrong rand_a rand_b; do
  for t in 1 2 3 4 5 6; do
    [ -f $D/$s/rft_config.json ] && break
    g=$(pick_gpu 9000); echo "=== [gpu$g] train $s try $t $(date)"
    CUDA_VISIBLE_DEVICES=$g $HPY -m train.rft_lora --gens $R/data/$s --model olmo2-1b-sft --label all --lr 1e-4 --epochs 2 --adapter_only $X --out $D/$s > $L/contrast_train_$s.log 2>&1 || { tail -1 $L/contrast_train_$s.log | cut -c1-160; sleep 120; }
  done
  [ -f $D/$s/rft_config.json ] || { echo "FAILED train $s"; exit 1; }
  grep "training solutions" $L/contrast_train_$s.log
done
ev () {  # name split terms...
  local n=$1 sp=$2; shift 2
  for t in 1 2 3 4 5 6; do
    [ -f $R/$n/$sp/$sp.metrics.json ] && break
    rm -rf $D/tmp; M=$D/tmp
    if [ "${1:-}" = INSTRUCT ]; then M=olmo2-1b-instruct; else $HPY -m analysis.lora_combo --model olmo2-1b-sft --terms "$@" --out $D/tmp > $L/contrast_build.log 2>&1 || { echo "FAILED build $n"; tail -1 $L/contrast_build.log; return; }; fi
    g=$(pick_gpu 11000)
    CUDA_VISIBLE_DEVICES=$g $HPY -m methods.run --method rl_rules --rl_beta 0 --model $M --format chat --split $sp $LIM --max_new_tokens 512 --batch_problems 16 --out $R/$n/$sp > $L/contrast_eval.log 2>&1 || { tail -1 $L/contrast_eval.log | cut -c1-160; sleep 120; }
  done
  [ -f $R/$n/$sp/$sp.metrics.json ] && echo "  $n $sp $(grep -oE '"acc_lenient": [0-9.]+' $R/$n/$sp/$sp.metrics.json) $(date +%T)" || echo "  FAILED $n $sp"; }
terms () { case $1 in
  sft) echo "" ;; instruct) echo INSTRUCT ;;
  rft_a*) echo "$D/gold_all:${1#rft_a}" ;;
  rftcon_l*) l=${1#rftcon_l}; echo "$D/gold_all:1 $D/correct:$l $D/wrong:-$l" ;;
  con_l*) l=${1#con_l}; echo "$D/correct:$l $D/wrong:-$l" ;;
  rftrand_l*) l=${1#rftrand_l}; echo "$D/gold_all:1 $D/rand_a:$l $D/rand_b:-$l" ;; esac; }
G="sft instruct rft_a1 rft_a1.5 rftcon_l0.25 rftcon_l0.5 rftcon_l1 con_l0.5 con_l1 con_l2 rftrand_l0.25 rftrand_l0.5 rftrand_l1"
[ $SMOKE = 1 ] && G="sft rftcon_l0.5"
for n in $G; do ev $n heldout500 $(terms $n); done
[ $SMOKE = 1 ] && exit 0
PICK=$($HPY - $R <<'PY'
import json, sys
R = sys.argv[1]; fam = {"rft_a": ["rft_a1", "rft_a1.5"], "rftcon": [f"rftcon_l{l}" for l in ("0.25", "0.5", "1")], "con": [f"con_l{l}" for l in ("0.5", "1", "2")], "rand": [f"rftrand_l{l}" for l in ("0.25", "0.5", "1")]}
out = ["sft", "instruct", "rft_a1"]
for f, ns in fam.items():
    v = {}
    for n in ns:
        try: v[n] = json.load(open(f"{R}/{n}/heldout500/heldout500.metrics.json"))["acc_lenient"]
        except Exception: pass
    if v: out.append(max(v, key=v.get))
print(" ".join(dict.fromkeys(out)))
PY
)
echo "full test: $PICK"
for n in $PICK; do ev $n test $(terms $n); done
$HPY - $R $PICK <<'PY'
import json, sys, os
from scipy.stats import binomtest
R = sys.argv[1]; X = {}
for n in sys.argv[2:]:
    p = f"{R}/{n}/test/test.generations.jsonl"
    if os.path.exists(p): X[n] = [(r["question"], r["paths"][r["chosen"]]["score"]["lenient_correct"]) for r in map(json.loads, open(p))]
for n, A in X.items():
    s = f"{n} {sum(c for _, c in A)/len(A):.4f}"
    for b in ("sft", "rft_a1", "instruct"):
        if b in X and b != n:
            B = X[b]; assert [q for q, _ in A] == [q for q, _ in B]
            x = sum(i and not j for (_, i), (_, j) in zip(A, B)); y = sum(j and not i for (_, i), (_, j) in zip(A, B)); s += f" | vs {b} {x}/{y} p={binomtest(x, max(1, x+y)).pvalue:.4f}"
    print(s)
PY
rm -rf $D/tmp; echo "=== done $(date)"
