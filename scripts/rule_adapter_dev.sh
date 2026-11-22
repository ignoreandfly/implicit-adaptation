#!/usr/bin/env bash
# Hewitt-style analysis for reasoning: ranking test (4 OLMo stages) + rule adapter on dev
# (best weights, small sweep, leave-one-out ablation). No prompt change: plain "Question/Answer".
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniconda3/envs/ICCV/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-1} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "=== ranking test $(date +%T)"; $PY -m analysis.ranking_test --n 500 2>&1 | grep -vi warn | tail -6
BEST="--ra_stop_pen 3 --ra_first_digit_pen 10 --ra_first_step_bonus 6"
m () { echo "=== $*  $(date +%T)"; $PY -m methods.run --method rule_adapter "$@" 2>&1 | grep -vi warn | tail -9; }
m $BEST
m $BEST --ra_step_bonus 0
m $BEST --ra_step_bonus 2
m --ra_stop_pen 3 --ra_first_digit_pen 10 --ra_first_step_bonus 9
for r in stop shift first rep; do m $BEST --ra_off $r; done
