#!/usr/bin/env bash
# Wait for the given PIDs (current GPU-1 queues), then sample SFT / DPO / Instruct with vLLM:
# n = 64, T = 0.7 (same temperature as the Base self-consistency runs), 500-problem test subset.
cd "$(dirname "$0")/.."
for p in "$@"; do while kill -0 "$p" 2>/dev/null; do sleep 30; done; done
SPECS="olmo2-1b-sft:0 olmo2-1b-instruct:0 olmo2-1b-dpo:0" N=64 TEMP=0.7 SUBSET=500 MEM=0.4 GPU=${GPU:-1} \
  bash scripts/gsm8k_sampling.sh
