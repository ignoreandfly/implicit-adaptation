#!/usr/bin/env bash
# Final comparison on the fixed 500-problem GSM8K test subset (data/gsm8k/test500.jsonl,
# same indices as evaluation.generate --subset 500). Base-model methods via methods.run;
# SFT/DPO/Instruct greedy numbers come from the existing full-test runs restricted to the subset.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniconda3/envs/ICCV/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-1} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COT=" Let's think step by step."
R=results/screen/olmo2-1b-base/test500
m () { echo "=== $*  $(date +%T)"; $PY -m methods.run --split test500 "$@" 2>&1 | grep -vi warn | tail -10; }
m --method greedy                                  # greedy zero-shot (no prefix)
m --method greedy --few_shot 8 --tag 8shot         # 8-shot prompting reference
m --method greedy --prefix "$COT"                  # zero-shot CoT prompt
m --method sample --n 10 --temperature 0.7 --prefix "$COT"   # self-consistency
echo "=== rescore $(date +%T)"; $PY -m methods.rescore $R/sample_prefix-let-s-think-step-by-step_T0.7_n10 --split test500 2>&1 | grep -vi warn | tail -30
