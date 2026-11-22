#!/usr/bin/env bash
# Fork-MBR on 7B SFT (test500), moved to GPU 1 AFTER GPU 1's other overnight jobs
# (7B Instruct runs, then cheap 1B FOBAR variants), so at most two GPUs are busy at once.
set -uo pipefail
cd "$(dirname "$0")/.."
until [ -f results/fobar/olmo2-1b-sft_sampling_k8_nb2m2a3.json ] && ! pgrep -f "overnight_cheap_fobar.sh" >/dev/null \
      && ! pgrep -f "olmo2-7b-instruct" >/dev/null; do sleep 60; done
export HF_HUB_CACHE=/data/azfarm/siddhant_hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=1
echo "=== fork-MBR 7B SFT test500 $(date)"
$HOME/miniconda3/envs/ICCV/bin/python -m methods.run --method fork_mbr --model olmo2-7b-sft --split test500 \
  --max_new_tokens 512 --fm_forks 8 --batch_problems 1 --fm_chunk 8 2>&1 | grep -vE "it/s" | tail -12
$HOME/miniconda3/envs/ICCV/bin/python scripts/overnight_summary.py > results/overnight_summary.md
git add results/overnight_summary.md results/screen/olmo2-7b-sft/test500/*/test500.metrics.json scripts/overnight_fork7b.sh
git commit -q -m "Fork-MBR on 7B SFT (test500); refreshed overnight summary

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q 2>&1 | tail -1
echo "=== done $(date)"
