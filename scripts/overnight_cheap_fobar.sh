#!/usr/bin/env bash
# Cheap FOBAR variants on 1B SFT pools (cost reduction), after the 7B Instruct runs free GPU 1.
# Re-writes the overnight summary and pushes at the end.
set -uo pipefail
cd "$(dirname "$0")/.."
until [ -f results/olmo2-7b-instruct/gsm8k/0shot_sub500_n64_t0.7/test.metrics.json ] && ! pgrep -f "olmo2-7b-instruct" >/dev/null; do sleep 60; done
VPY=$HOME/miniconda3/envs/implicit-vllm/bin/python
export CUDA_VISIBLE_DEVICES=1
for pool in fork sampling; do
  for cfg in "--n_back 1 --masks 1 --max_answers 2" "--n_back 1 --masks 2 --max_answers 2" "--n_back 2 --masks 2 --max_answers 3"; do
    tag=$(echo "$cfg" | tr -d ' -' | sed 's/n_back/nb/;s/masks/m/;s/max_answers/a/')
    echo "=== $pool $cfg $(date)"
    $VPY -m analysis.fobar --pool $pool --k 8 $cfg --out results/fobar/olmo2-1b-sft_${pool}_k8_${tag}.json 2>&1 | grep -E '"(majority|backward|fobar|oracle|rescue_fobar|harm_fobar|backward_prompts)"'
  done
done
~/miniconda3/envs/ICCV/bin/python scripts/overnight_summary.py > results/overnight_summary.md
git add results/overnight_summary.md results/fobar/*.json scripts/overnight_cheap_fobar.sh
git commit -q -m "Cheap FOBAR variants on 1B SFT pools; refreshed overnight summary

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q 2>&1 | tail -1
