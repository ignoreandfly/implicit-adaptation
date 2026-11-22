#!/usr/bin/env bash
# Second screening batch on OLMo 2 1B Base (dev, 200): decoders on top of the zero-shot CoT prefix.
# gamma = 1.0 / beta = 0.0 are each method's own controls (same batch layout => same bf16 numerics).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniconda3/envs/ICCV/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-1} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COT=" Let's think step by step."
m () { echo "=== $*  $(date +%T)"; $PY -m methods.run "$@" --prefix "$COT" 2>&1 | grep -vi warn | tail -10; }
for g in 1.0 1.25 1.5 2.0; do m --method cfg --cfg_gamma $g; done
for b in 0.0 0.25 0.5; do m --method contrastive --cd_beta $b --dropout 0.2; done
