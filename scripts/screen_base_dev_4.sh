#!/usr/bin/env bash
# Fourth screening batch (dev, 200): Power-SMC on top of zero-shot CoT, then re-run rescoring
# (now incl. embedding-based agreement) on all multi-path runs. Waits for the given PIDs.
set -uo pipefail
cd "$(dirname "$0")/.."
for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 20; done; done
PY=${PY:-$HOME/miniconda3/envs/ICCV/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-1} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COT=" Let's think step by step."
R=results/screen/olmo2-1b-base/dev
m () { echo "=== $*  $(date +%T)"; $PY -m methods.run "$@" --batch_problems 4 --prefix "$COT" 2>&1 | grep -vi warn | tail -12; }
m --method power_smc --n 10 --smc_alpha 4
m --method power_smc --n 16 --smc_alpha 4
for d in $R/sample_prefix-* $R/power_smc_* $R/eager_* $R/cot_decoding_k10; do
  [ -f $d/dev.metrics.json ] || continue
  echo "=== rescore $d $(date +%T)"; $PY -m methods.rescore $d 2>&1 | grep -vi warn | tail -30
done
