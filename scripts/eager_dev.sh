#!/usr/bin/env bash
# EAGer on dev (re-run after fixing the entropy line), then rescoring.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniconda3/envs/ICCV/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-1} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COT=" Let's think step by step."
R=results/screen/olmo2-1b-base/dev
for th in 1.0 1.5 2.0; do echo "=== eager theta $th $(date +%T)"; $PY -m methods.run --method eager --n 10 --eager_theta $th --batch_problems 4 --prefix "$COT" 2>&1 | grep -vi warn | tail -10; done
for d in $R/eager_*; do echo "=== rescore $d $(date +%T)"; $PY -m methods.rescore $d 2>&1 | grep -vi warn | tail -30; done
