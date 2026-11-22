#!/usr/bin/env bash
# Rule adapter, round 3 (dev): ablation showed "first" carries the gain and word-level rep / shift hurt.
# Try minimal rule sets. Waits for given PIDs.
set -uo pipefail
cd "$(dirname "$0")/.."
for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 15; done; done
PY=${PY:-$HOME/miniconda3/envs/ICCV/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-1} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
F="--ra_stop_pen 3 --ra_first_digit_pen 10 --ra_first_step_bonus 6"
m () { echo "=== $*  $(date +%T)"; $PY -m methods.run --method rule_adapter --batch_problems 4 "$@" 2>&1 | grep -vi warn | tail -9; }
m $F --ra_off shift,rep,stop                                   # first only
m $F --ra_off shift,stop --ra_rep_pen 0                          # first + 4-gram loop penalty
m $F --ra_off shift --ra_rep_pen 0 --ra_min_len 30               # first + loop penalty + short stop block
m $F --ra_off shift,rep --ra_min_len 30                          # first + short stop block
