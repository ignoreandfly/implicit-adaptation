#!/usr/bin/env bash
# Full test (1,319) for the existing SFT RFT models (gold / label-free) and the SFT baseline, all in one pipeline
# (rl_rules --rl_beta 0, chat, batch 2, 512 tokens). Instruct in this pipeline: results/weight_sweep/dpo__instruct_fulltest_a1.
# Waits for >= 8 GB free on any GPU; retries on crashes; then paired McNemar tests.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HPY=$HOME/miniconda3/envs/ICCV/bin/python; D=/data/azfarm/siddhant_fork_lora; R=results/sft_rft_fulltest; L=logs/queue
mkdir -p $R
pick_gpu () { while true; do g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
  awk -F', ' '$2 >= 8000 {print $2, $1}' | sort -rn | awk 'NR==1{print $2}'); [ -n "$g" ] && { echo $g; return; }; sleep 60; done; }
ev () {  # tag model
  for t in $(seq 1 20); do
    [ -f $R/$1/test.metrics.json ] && break
    g=$(pick_gpu); echo "=== [gpu$g] $1 try $t $(date)"
    CUDA_VISIBLE_DEVICES=$g $HPY -m methods.run --method rl_rules --rl_beta 0 --model $2 --format chat --split test \
      --max_new_tokens 512 --batch_problems 2 --out $R/$1 > $L/sft_rft_full_$1.log 2>&1 || { tail -1 $L/sft_rft_full_$1.log | cut -c1-200; sleep 180; }
  done
  [ -f $R/$1/test.metrics.json ] && echo "  $1 $(grep -oE '"acc_lenient": [0-9.]+' $R/$1/test.metrics.json)" || echo "  $1 FAILED"
}
ev sft olmo2-1b-sft
ev sft_rft_gold $D/sft_rft_gold
ev sft_rft_majority $D/sft_rft_majority
$HPY - <<'PY'
import json, os
from scipy.stats import binomtest
def L(p): return [(r["question"], r["paths"][r["chosen"]]["score"]["lenient_correct"]) for r in map(json.loads, open(p))]
P = {t: f"results/sft_rft_fulltest/{t}/test.generations.jsonl" for t in ("sft", "sft_rft_gold", "sft_rft_majority")}
P["instruct"] = "results/weight_sweep/dpo__instruct_fulltest_a1/alpha_1/test.generations.jsonl"
X = {k: L(p) for k, p in P.items() if os.path.exists(p)}
for a, b in (("sft_rft_gold", "sft"), ("sft_rft_majority", "sft"), ("sft_rft_gold", "sft_rft_majority"), ("instruct", "sft_rft_gold"), ("instruct", "sft")):
    if a not in X or b not in X: print("missing", a, b); continue
    A, B = X[a], X[b]; assert [q for q, _ in A] == [q for q, _ in B] and len(A) == 1319
    x = sum(i and not j for (_, i), (_, j) in zip(A, B)); y = sum(j and not i for (_, i), (_, j) in zip(A, B))
    print(f"{a} {sum(c for _, c in A)/len(A):.4f} vs {b} {sum(c for _, c in B)/len(B):.4f}: {x}/{y} p={binomtest(x, x+y).pvalue:.4f}")
PY
echo "=== done $(date)"
