#!/usr/bin/env bash
# Screen training-free methods on OLMo 2 1B Base, zero-shot, dev split (200 problems).
#   GPU=1 bash scripts/screen_base_dev.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniconda3/envs/ICCV/bin/python}
export CUDA_VISIBLE_DEVICES=${GPU:-1} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
COT=" Let's think step by step."
R=results/screen/olmo2-1b-base/dev
m () { echo "=== $*  $(date +%T)"; $PY -m methods.run "$@" 2>&1 | grep -vi warn | tail -14; }
# single path
m --method greedy --prefix "$COT"
m --method greedy --prefix " First,"
m --method greedy --prefix " Let's"
m --method greedy --rep_penalty 1.2
m --method dola --dola_layers high --rep_penalty 1.2
m --method dola --dola_layers low --rep_penalty 1.2
m --method dola --dola_layers high --rep_penalty 1.0
m --method budget_forcing --prefix "$COT" --rounds 1
# multi path (n = 10, same budget as CoT-decoding k = 10)
m --method sample --n 10 --temperature 0.7
m --method sample --n 10 --temperature 0.7 --prefix "$COT"
m --method sample --n 10 --prefix "$COT" --fork_threshold 1.0 --t_low 0.3 --t_high 1.0
m --method sample --n 10 --prefix "$COT" --fork_threshold 2.0 --t_low 0.3 --t_high 1.0
# selection rules on every multi-path run
for d in $R/cot_decoding_k10 $R/sample_*; do echo "=== rescore $d $(date +%T)"; $PY -m methods.rescore $d 2>&1 | grep -vi warn | tail -22; done
