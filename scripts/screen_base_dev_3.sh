#!/usr/bin/env bash
# Third screening batch (dev, 200): EAGer and thought-level self-certainty on top of zero-shot CoT,
# then rescoring of the new multi-path runs. Waits for the PIDs given as arguments to finish.
set -uo pipefail
cd "$(dirname "$0")/.."
for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 20; done; done
PY=${PY:-$HOME/miniconda3/envs/ICCV/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-1} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COT=" Let's think step by step."
R=results/screen/olmo2-1b-base/dev
m () { echo "=== $*  $(date +%T)"; $PY -m methods.run "$@" --batch_problems 4 --prefix "$COT" 2>&1 | grep -vi warn | tail -10; }
m --method tlsc --tlsc_k 1 --tag K1_control              # greedy control in the same thought-by-thought loop
m --method tlsc --tlsc_k 4 --tlsc_steps 40
m --method tlsc --tlsc_k 4 --tlsc_steps 3
m --method eager --n 10 --eager_theta 2.0
m --method eager --n 10 --eager_theta 2.5
for d in $R/eager_*; do echo "=== rescore $d $(date +%T)"; $PY -m methods.rescore $d 2>&1 | grep -vi warn | tail -26; done
