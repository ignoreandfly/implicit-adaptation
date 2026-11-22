#!/usr/bin/env bash
# Pairwise self-judge on OLMo 2 1B SFT: (1) 8 SFT samples on 1,500 GSM8K training questions (HF sampling),
# (2) train the judge LoRA on (correct, incorrect) pairs of its own solutions, (3) selection on the saved
# test500 pools (k = 4, 8, 16) vs majority vote, for the untrained (zero-shot) and the trained judge.
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HPY=$HOME/miniconda3/envs/ICCV/bin/python; L=logs/queue; mkdir -p results/judge
GENS=results/judge/olmo2-1b-sft_train1500_n8; JUDGE=/data/azfarm/siddhant_fork_lora/judge_sft
pick_gpu () { while true; do g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
  awk -F', ' -v need="$1" '$2 >= need {print $1; exit}'); [ -n "$g" ] && { echo "$g"; return; }; sleep 90; done; }
for try in 1 2 3; do
  [ -f $GENS/train.generations.jsonl ] && break
  g=$(pick_gpu 6000); echo "=== [gpu$g] SFT samples on 1,500 training questions (try $try) $(date)"
  CUDA_VISIBLE_DEVICES=$g $HPY -m evaluation.generate --model olmo2-1b-sft --split train --limit 1500 --n_samples 8 \
    --temperature 0.7 --batch_size 32 --max_new_tokens 512 --out $GENS > $L/judge_sample.log 2>&1
done
[ -f $GENS/train.generations.jsonl ] || { echo "sampling failed"; exit 1; }
g=$(pick_gpu 6000); echo "=== [gpu$g] zero-shot judge selection (untrained SFT) $(date)"
CUDA_VISIBLE_DEVICES=$g $HPY -m analysis.judge_select --model olmo2-1b-sft --judge base --k 4 8 16 2>&1 | grep -E "^k=" 
g=$(pick_gpu 8000); echo "=== [gpu$g] train judge $(date)"
CUDA_VISIBLE_DEVICES=$g $HPY -m train.judge_lora --gens $GENS --model olmo2-1b-sft --out $JUDGE > $L/judge_train.log 2>&1
grep -E "pairs from|accuracy" $L/judge_train.log
cp $JUDGE/judge_config.json results/judge/ 2>/dev/null
g=$(pick_gpu 6000); echo "=== [gpu$g] trained judge selection $(date)"
CUDA_VISIBLE_DEVICES=$g $HPY -m analysis.judge_select --model olmo2-1b-sft --judge $JUDGE --k 4 8 16 2>&1 | grep -E "^k="
git add results/judge/*.json train/judge_lora.py analysis/judge_select.py scripts/queue_judge.sh 2>/dev/null
git commit -q -m "Pairwise self-judge on 1B SFT: training + selection vs majority on test500 pools

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q 2>&1 | tail -1
echo "=== done $(date)"
