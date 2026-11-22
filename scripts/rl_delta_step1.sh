#!/usr/bin/env bash
# Step 1 of the "sparse RL rules" plan: greedy SFT / DPO / Instruct solutions on the first 300
# GSM8K TRAINING problems (never the test set), then token-level distribution comparisons
# SFT->Instruct (all post-SFT training), SFT->DPO and DPO->Instruct (the RLVR stage alone).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniconda3/envs/ICCV/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-1} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for m in olmo2-1b-sft olmo2-1b-dpo olmo2-1b-instruct; do
  out=results/rl_delta/${m}_train300
  [ -f $out/train.metrics.json ] || { echo "=== generate $m $(date +%T)"; $PY -m evaluation.generate --model $m --split train --limit 300 --batch_size 32 --out $out 2>&1 | grep -vi warn | tail -2; }
done
d () { echo "=== delta $1 -> $2 on $3 trajectories $(date +%T)"; $PY -m analysis.rl_delta --gens results/rl_delta/$3_train300 --a $1 --b $2 2>&1 | grep -vi warn | tail -45; }
d olmo2-1b-sft olmo2-1b-instruct olmo2-1b-sft
d olmo2-1b-dpo olmo2-1b-instruct olmo2-1b-dpo
d olmo2-1b-sft olmo2-1b-dpo olmo2-1b-sft
