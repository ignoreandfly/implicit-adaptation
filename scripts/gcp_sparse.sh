#!/usr/bin/env bash
# Sparse extrapolation of the RL step (analysis/sparse_extrap.py) on the full GSM8K test, OLMo 2 1B.
#   SHUTDOWN=1 bash scripts/gcp_sparse.sh       (VPY = python of the vllm env)
set -uo pipefail
cd "$(dirname "$0")/.."
VPY=${VPY:-python}; mkdir -p logs
$VPY -m analysis.sparse_extrap --out results/sparse_extrap --configs \
  dense:1 dense:1.5 \
  top:0.01:1.5 top:0.05:1.5 top:0.2:1.5 \
  rand:0.05:1.5 rand:0.2:1.5 \
  top:0.05:1 \
  > logs/sparse_extrap.log 2>&1
rm -rf checkpoints/sparse_extrap_tmp
echo "=== gcp_sparse done $(date)" >> logs/sparse_extrap.log
[ "${SHUTDOWN:-0}" = 1 ] && sudo shutdown -h +2 "gcp_sparse finished"
