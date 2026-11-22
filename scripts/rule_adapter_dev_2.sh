#!/usr/bin/env bash
# Rule adapter, round 2: stop rule only blocks stopping during the first --ra_min_len tokens
# (closer to Hewitt's EOS rule, which never blocks stopping). Dev only. Waits for given PIDs.
set -uo pipefail
cd "$(dirname "$0")/.."
for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 15; done; done
PY=${PY:-$HOME/miniconda3/envs/ICCV/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-1} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
B="--ra_stop_pen 3 --ra_first_digit_pen 10 --ra_first_step_bonus 6 --ra_step_bonus 0"
m () { echo "=== $*  $(date +%T)"; $PY -m methods.run --method rule_adapter --batch_problems 4 "$@" 2>&1 | grep -vi warn | tail -9; }
for ml in 10 30 60; do m $B --ra_min_len $ml; done
