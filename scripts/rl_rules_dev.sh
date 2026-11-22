#!/usr/bin/env bash
# Track B on dev (200): SFT + RL-derived rules. Control, oracle, swap rules, gentle bias tables.
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniconda3/envs/ICCV/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-1} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
m () { echo "=== $*  $(date +%T)"; $PY -m methods.run --method rl_rules --model olmo2-1b-sft --max_new_tokens 512 --batch_problems 4 "$@" 2>&1 | grep -vi warn | grep -E "swap rules|=== |acc_lenient|mean_gen|trunc|share|Error" | tail -6; }
m --rl_beta 0
m --rl_oracle
for M in 10 20 40; do m --rl_swaps $M; done
for b in 0.25 0.5; do m --rl_k 10 --rl_beta $b; done
m --rl_oracle_model olmo2-1b-instruct --rl_oracle --rl_scope all --tag instruct_greedy_ref   # = Instruct greedy on dev
