#!/usr/bin/env bash
# 7B experiments for a dedicated cloud GPU (see RUNBOOK.md, section 6). Run from the repo root, inside the
# environments described in RUNBOOK.md section 1. Each step is skipped if its main output already exists,
# so the script can be re-run after an interruption.
#   JOBS="sample rft judge fobar forkmbr" bash scripts/gcp_7b_jobs.sh      (default: all, in this order)
#   VPY=... HPY=...   paths to the vllm-env and hf-env pythons (default: python in PATH for both)
set -uo pipefail
cd "$(dirname "$0")/.."
VPY=${VPY:-python}; HPY=${HPY:-python}
JOBS=${JOBS:-"sample rft judge fobar forkmbr"}
MEM=${MEM:-0.85}                         # vLLM memory share (dedicated GPU)
OUT=${OUT:-checkpoints}; mkdir -p $OUT logs
COT=" Let's think step by step."
POOL=results/olmo2-7b-sft/gsm8k/0shot_sub500_n64_t0.7     # test500 pool (n = 64) for selection experiments
TRAIN=results/judge/olmo2-7b-sft_train1500_n8             # 8 samples on 1,500 training questions
has () { [[ " $JOBS " == *" $1 "* ]]; }
acc () { grep -oE "'?\"?acc_lenient\"?'?: [0-9.]+" $1 | tail -1; }

# 0. data
[ -f data/gsm8k/test500.jsonl ] || {
  $HPY -m tasks.gsm8k --out data/gsm8k
  $HPY -c "
from evaluation.generate import subset_indices
from common.io import read_jsonl, write_jsonl
rows = read_jsonl('data/gsm8k/test.jsonl'); idx = subset_indices(len(rows), 500, 0)
write_jsonl('data/gsm8k/test500.jsonl', [{**rows[i], 'idx': i} for i in idx])"; }

if has sample; then   # ~30-40 min each on an L4
  [ -f $POOL/test.generations.jsonl ] || $VPY -m evaluation.generate --backend vllm --model olmo2-7b-sft --subset 500 \
     --n_samples 64 --temperature 0.7 --gpu_memory_utilization $MEM --out $POOL > logs/gcp_pool.log 2>&1
  [ -f $TRAIN/train.generations.jsonl ] || $VPY -m evaluation.generate --backend vllm --model olmo2-7b-sft --split train \
     --limit 1500 --n_samples 8 --temperature 0.7 --gpu_memory_utilization $MEM --out $TRAIN > logs/gcp_train_samples.log 2>&1
  echo "sample: pool $(ls $POOL) | train $(ls $TRAIN)"
fi

if has rft; then      # strongest 1B method at 7B: fine-tune 7B SFT on its own correct / majority-agreeing samples
  for lab in gold majority; do
    tag=sft7b_rft_$lab
    [ -f $OUT/$tag/model.safetensors ] || $HPY -m train.rft_lora --gens $TRAIN --split train --model olmo2-7b-sft \
       --label $lab --lr 1e-4 --epochs 2 --out $OUT/$tag > logs/gcp_train_$tag.log 2>&1
    mkdir -p results/rft_sft7b/$tag; cp $OUT/$tag/rft_config.json results/rft_sft7b/$tag/ 2>/dev/null
    $VPY -m evaluation.generate --backend vllm --model $OUT/$tag --format chat --subset 500 --gpu_memory_utilization $MEM \
       --out results/rft_sft7b/$tag/greedy_test500 > logs/gcp_eval_$tag.log 2>&1
    echo "rft $lab test500: $(acc logs/gcp_eval_$tag.log)   (7B SFT greedy 0.746, 7B Instruct greedy 0.832)"
  done
fi

if has judge; then    # does a trained self-judge beat majority vote at 7B?
  $HPY -m analysis.judge_select --model olmo2-7b-sft --judge base --pool $POOL/test.generations.jsonl --k 4 8 16 \
     --out results/judge/select_olmo2-7b-sft_base.json 2>&1 | grep -E "^k=" | sed 's/^/zero-shot judge /'
  [ -f $OUT/judge_sft7b/model.safetensors ] || $HPY -m train.judge_lora --gens $TRAIN --model olmo2-7b-sft \
     --out $OUT/judge_sft7b > logs/gcp_judge_train.log 2>&1
  grep -E "accuracy" logs/gcp_judge_train.log; cp $OUT/judge_sft7b/judge_config.json results/judge/judge_config_7b.json 2>/dev/null
  $HPY -m analysis.judge_select --model olmo2-7b-sft --judge $OUT/judge_sft7b --pool $POOL/test.generations.jsonl --k 4 8 16 \
     --out results/judge/select_olmo2-7b-sft_trained.json 2>&1 | grep -E "^k=" | sed 's/^/trained judge /'
fi

if has fobar; then    # backward verification on 7B sample pools
  for k in 8 16; do
    $VPY -m analysis.fobar --model olmo2-7b-sft --pool sampling --k $k --gpu_memory_utilization $MEM \
       --sampling_gens $POOL/test.generations.jsonl --out results/fobar/olmo2-7b-sft_sampling_k$k.json 2>&1 | \
       grep -E '"(majority|fobar|rescue_fobar|harm_fobar)"' | tr -d '\n '; echo " (fobar k=$k)"
  done
fi

if has forkmbr; then  # fork-gated MBR on 7B SFT (HF, ~3-5 h on an L4)
  $HPY -m methods.run --method fork_mbr --model olmo2-7b-sft --split test500 --max_new_tokens 512 --fm_forks 8 \
     --batch_problems 1 --fm_chunk 8 > logs/gcp_forkmbr7b.log 2>&1
  grep -E '"acc_(greedy|F[0-9])"' logs/gcp_forkmbr7b.log | tr -d '\n '; echo " (fork-MBR 7B)"
fi
echo "=== gcp_7b_jobs done $(date)"
