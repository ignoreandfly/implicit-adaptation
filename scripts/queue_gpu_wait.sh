#!/usr/bin/env bash
# Queue (2026-11-23): two lanes, each job waits until SOME GPU has enough free memory, then runs there.
# No process-name waits (they kept matching their own launcher shells). Results committed + pushed at the end.
#   lane A (1B): fork-only inference with round-3 LoRAs (dev) -> fork pairs on ALL 7,273 training
#                questions -> retrain (majority / gold / shuffled) on the full data, dev eval
#   lane B (7B): FOBAR on 7B SFT sampling pools (k = 8, 16) -> fork-MBR on 7B SFT (test500)
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_CACHE_7B=/data/azfarm/siddhant_hf_cache
VPY=$HOME/miniconda3/envs/implicit-vllm/bin/python; HPY=$HOME/miniconda3/envs/ICCV/bin/python
OUT=/data/azfarm/siddhant_fork_lora; R=results/fork_lora
mkdir -p logs/queue

pick_gpu () {  # $1 = MiB needed; prints a GPU index once one has that much free (checks every 2 min)
  while true; do
    g=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
        awk -F', ' -v need="$1" '$2 >= need {print $1; exit}')
    [ -n "$g" ] && { echo "$g"; return; }
    sleep 120
  done
}

laneA () {
  # A1: fork-only inference (SFT decodes; at forks, i.e. SFT entropy above the train 80th percentile,
  #     the token comes from the LoRA model). Control: plain SFT through the same decoder.
  for tag in sft_control dpo_major_lr1e-4_e2 dpo_gold_lr1e-4_e2 dpo_shuffled_lr1e-4_e2 dpo_major_lr3e-4_e2; do
    g=$(pick_gpu 9000); echo "=== [A1 gpu$g] fork-only inference $tag $(date)"
    if [ "$tag" = sft_control ]; then extra="--rl_beta 0"; else extra="--rl_oracle --rl_oracle_model $OUT/$tag"; fi
    CUDA_VISIBLE_DEVICES=$g $HPY -m methods.run --method rl_rules --model olmo2-1b-sft --split dev --max_new_tokens 512 \
      --batch_problems 4 $extra --tag forkonly_$tag > logs/queue/forkonly_$tag.log 2>&1
    grep -oE '"acc_lenient": [0-9.]+' logs/queue/forkonly_$tag.log | tail -1 | sed "s/^/  /"
  done
  # A2: fork pairs on all training questions
  PAIRS=data/fork_pairs/olmo2-1b-sft_trainall.jsonl
  if [ ! -f $PAIRS ]; then
    g=$(pick_gpu 9000); echo "=== [A2 gpu$g] fork pairs, all 7,273 training questions $(date)"
    CUDA_VISIBLE_DEVICES=$g $VPY -m train.fork_pairs --model olmo2-1b-sft --n 7273 --gpu_memory_utilization 0.3 \
      --out $PAIRS > logs/queue/fork_pairs_all.log 2>&1
    tail -1 logs/queue/fork_pairs_all.log
  fi
  # A3: retrain on the full data, dev eval
  for spec in "full_dpo_major_lr1e-4:--label majority" "full_dpo_gold_lr1e-4:--label gold" "full_dpo_shuffled_lr1e-4:--label majority --shuffle"; do
    tag=${spec%%:*}; args=${spec#*:}
    g=$(pick_gpu 10000); echo "=== [A3 gpu$g] train $tag $(date)"
    CUDA_VISIBLE_DEVICES=$g $HPY -m train.fork_lora --pairs $PAIRS --model olmo2-1b-sft --loss dpo --lr 1e-4 --beta 0.5 \
      --epochs 1 $args --out $OUT/$tag > logs/queue/train_$tag.log 2>&1
    grep -E "training forks" logs/queue/train_$tag.log | tail -1
    CUDA_VISIBLE_DEVICES=$g $VPY -m evaluation.generate --backend vllm --model $OUT/$tag --format chat --split dev \
      --gpu_memory_utilization 0.3 --out $R/$tag/greedy_dev > logs/queue/eval_$tag.log 2>&1
    grep -oE "'acc_lenient': [0-9.]+" logs/queue/eval_$tag.log | tail -1 | sed "s/^/  $tag dev greedy /"
    mkdir -p $R/$tag; cp $OUT/$tag/fork_lora_config.json $R/$tag/ 2>/dev/null
  done
}

laneB () {
  for k in 8 16; do
    g=$(pick_gpu 21000); echo "=== [B gpu$g] FOBAR 7B SFT k=$k $(date)"
    CUDA_VISIBLE_DEVICES=$g HF_HUB_CACHE=$HF_HUB_CACHE_7B $VPY -m analysis.fobar --model olmo2-7b-sft --pool sampling --k $k \
      --gpu_memory_utilization 0.8 --sampling_gens results/olmo2-7b-sft/gsm8k/0shot_sub500_n64_t0.7/test.generations.jsonl \
      --out results/fobar/olmo2-7b-sft_sampling_k$k.json > logs/queue/fobar7b_k$k.log 2>&1
    grep -E '"(majority|fobar|rescue_fobar|harm_fobar)"' logs/queue/fobar7b_k$k.log | tr -d '\n '; echo
  done
  g=$(pick_gpu 20000); echo "=== [B gpu$g] fork-MBR 7B SFT test500 $(date)"
  CUDA_VISIBLE_DEVICES=$g HF_HUB_CACHE=$HF_HUB_CACHE_7B $HPY -m methods.run --method fork_mbr --model olmo2-7b-sft \
    --split test500 --max_new_tokens 512 --fm_forks 8 --batch_problems 1 --fm_chunk 8 > logs/queue/forkmbr7b.log 2>&1
  grep -E '"acc_(greedy|F[0-9])"' logs/queue/forkmbr7b.log | tr -d '\n '; echo
}

laneA > logs/queue/laneA.log 2>&1 &
A=$!
sleep 300   # stagger so both lanes do not grab the same freshly-freed GPU
laneB > logs/queue/laneB.log 2>&1 &
B=$!
wait $A $B
$HPY scripts/overnight_summary.py > results/overnight_summary.md 2>/dev/null
git add results/overnight_summary.md results/fobar/*.json results/fork_lora results/screen/*/*/*/*.metrics.json scripts/queue_gpu_wait.sh 2>/dev/null
git commit -q -m "Queue results: fork-only inference with LoRA, full-data fork-LoRA, FOBAR 7B, fork-MBR 7B

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q 2>&1 | tail -1
echo "=== queue done $(date)"
