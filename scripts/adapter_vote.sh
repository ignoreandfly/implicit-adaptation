#!/usr/bin/env bash
# Track A: Base + rule adapter (+ voting). Two dev-tuned rule sets, each with 10 samples + majority
# vote on dev; the one with higher dev vote accuracy is then run on test500 (greedy and voting).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniconda3/envs/ICCV/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-2} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A="--ra_stop_pen 3 --ra_first_digit_pen 10 --ra_first_step_bonus 6 --ra_off rep"
B="--ra_stop_pen 3 --ra_first_digit_pen 10 --ra_first_step_bonus 6 --ra_step_bonus 0"
V="--ra_n 10 --temperature 0.7 --batch_problems 4"
m () { echo "=== $*  $(date +%T)"; $PY -m methods.run --method rule_adapter "$@" 2>&1 | grep -vi warn | tail -10; }
m $A $V --tag A_vote; m $B $V --tag B_vote
best=$($PY -c "
import json
a=json.load(open('results/screen/olmo2-1b-base/dev/rule_adapter_A_vote/dev.metrics.json'))['acc_lenient']
b=json.load(open('results/screen/olmo2-1b-base/dev/rule_adapter_B_vote/dev.metrics.json'))['acc_lenient']
print('A' if a >= b else 'B')")
echo "=== dev-selected rule set: $best"
[ "$best" = A ] && CFG="$A" || CFG="$B"
m $CFG --split test500 --tag ${best}_greedy
m $CFG $V --split test500 --tag ${best}_vote
