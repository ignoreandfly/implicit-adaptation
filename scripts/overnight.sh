#!/usr/bin/env bash
# Overnight queue (2026-11-22/23). Three GPU lanes, each waits for the job currently on its GPU:
#   GPU 2: wait 7B SFT n=64  -> FOBAR on 7B SFT sampling pools (k = 8, 16)
#   GPU 1: wait 1B FOBAR (fork pool) -> 7B Instruct greedy + n = 64 (test500)
#   GPU 0: wait 1B FOBAR (sampling k=16) -> fork-MBR on 7B SFT (test500, F = 8)
# Then: results/overnight_summary.md, commit + push.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HUB_CACHE=/data/azfarm/siddhant_hf_cache PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VPY=$HOME/miniconda3/envs/implicit-vllm/bin/python
HPY=$HOME/miniconda3/envs/ICCV/bin/python
mkdir -p logs

wait_for () {  # $1 = file that must exist, $2 = process pattern that must be gone
  until [ -f "$1" ] && ! pgrep -f "$2" >/dev/null; do sleep 60; done
}

lane2 () {
  wait_for results/olmo2-7b-sft/gsm8k/0shot_sub500_n64_t0.7/test.metrics.json "scripts/olmo7b_sft.sh"
  for k in 8 16; do
    echo "=== [GPU2] FOBAR 7B SFT k=$k $(date)"
    CUDA_VISIBLE_DEVICES=2 $VPY -m analysis.fobar --model olmo2-7b-sft --pool sampling --k $k --gpu_memory_utilization 0.8 \
      --sampling_gens results/olmo2-7b-sft/gsm8k/0shot_sub500_n64_t0.7/test.generations.jsonl \
      --out results/fobar/olmo2-7b-sft_sampling_k$k.json 2>&1 | grep -vE "INFO|WARN|it/s" | tail -25
  done
}

lane1 () {
  wait_for results/fobar/olmo2-1b-sft_fork_k8.json "analysis.fobar --pool fork"
  echo "=== [GPU1] 7B Instruct greedy $(date)"
  CUDA_VISIBLE_DEVICES=1 $VPY -m evaluation.generate --backend vllm --model olmo2-7b-instruct --subset 500 \
    --gpu_memory_utilization 0.8 --out results/olmo2-7b-instruct/gsm8k/0shot_sub500_greedy 2>&1 | grep -vE "INFO|WARN" | tail -3
  echo "=== [GPU1] 7B Instruct n=64 $(date)"
  CUDA_VISIBLE_DEVICES=1 $VPY -m evaluation.generate --backend vllm --model olmo2-7b-instruct --subset 500 --n_samples 64 \
    --temperature 0.7 --gpu_memory_utilization 0.8 --out results/olmo2-7b-instruct/gsm8k/0shot_sub500_n64_t0.7 2>&1 | grep -vE "INFO|WARN" | tail -3
}

lane0 () {
  wait_for results/fobar/olmo2-1b-sft_sampling_k16.json "analysis.fobar --pool sampling"
  echo "=== [GPU0] fork-MBR 7B SFT test500 $(date)"
  CUDA_VISIBLE_DEVICES=0 $HPY -m methods.run --method fork_mbr --model olmo2-7b-sft --split test500 --max_new_tokens 512 \
    --fm_forks 8 --batch_problems 1 --fm_chunk 8 2>&1 | grep -vE "it/s" | tail -12
}

lane2 > logs/overnight_gpu2.log 2>&1 &
P2=$!
lane1 > logs/overnight_gpu1.log 2>&1 &
P1=$!
lane0 > logs/overnight_gpu0.log 2>&1 &
P0=$!
wait $P0 $P1 $P2

echo "=== summary $(date)"
$HPY scripts/overnight_summary.py > results/overnight_summary.md 2> logs/overnight_summary.err
git add results/overnight_summary.md results/fobar/*.json results/olmo2-7b-*/gsm8k/*/test.metrics.json \
        results/screen/*/test500/*/test500.metrics.json scripts/ methods/ analysis/ 2>/dev/null
git commit -q -m "Overnight runs: 7B SFT/Instruct sampling, FOBAR (1B, 7B), fork-MBR 7B; summary in results/overnight_summary.md

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q 2>&1 | tail -1
echo "=== done $(date)"
