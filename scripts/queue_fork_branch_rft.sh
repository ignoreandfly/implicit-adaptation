#!/usr/bin/env bash
# Fork-branch RFT on OLMo 2 1B Base + CoT: use the model's own forks to create correct training solutions
# (greedy decode -> branch at the 4 smallest top-2 margins with the 2nd token -> continue greedily -> keep correct ones),
# then plain RFT (positive-only, all tokens). Anchor: RFT-686 on 8 samples of the same 1,000 questions (43.7% full test).
#   fbr_only : RFT on greedy + fork branches only (5 paths / question)
#   fbr_union: RFT on the 8 samples + the fork branches
# Same trainer (train/rft_lora.py, lr 1e-4, 2 epochs), final model, greedy full test (same command as RFT-686).
set -uo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
HPY=$HOME/miniconda3/envs/ICCV/bin/python; VPY=$HOME/miniconda3/envs/implicit-vllm/bin/python
OUT=/data/azfarm/siddhant_fork_lora; R=results/fork_branch_rft; L=logs/queue; GPU=${GPU:?set GPU}
COT=" Let's think step by step."; M=olmo2-1b-base
SAMPLES=results/screen/olmo2-1b-base/train/sample_prefix-let-s-think-step-by-step_T0.7_n8
BR=results/fork_branch/base_cot_train1000_F4
mkdir -p $R
wait_mem () { while true; do f=$(nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ');
  [ "$f" -ge "$1" ] && return; sleep 60; done; }
acc () { grep -oE '"acc_lenient": [0-9.]+' $1 | tail -1 | grep -oE '[0-9.]+$'; }
# 1. fork-branch data (vLLM; retries for start-up failures on the shared GPU)
for t in 1 2 3 4; do
  [ -f $BR/train.generations.jsonl ] && break
  wait_mem 9000; echo "=== [gpu$GPU] fork-branch data, try $t $(date)"
  CUDA_VISIBLE_DEVICES=$GPU $VPY -m train.fork_branch_data --model $M --format plain --prefix "$COT" --n 1000 \
    --forks 4 --m 0 --n_label 8 --max_tokens 320 --gpu_memory_utilization 0.3 --out $BR > $L/fbr_data.log 2>&1
  tail -1 $L/fbr_data.log; sleep 30
done
[ -f $BR/train.generations.jsonl ] || { echo "branch data failed: $L/fbr_data.log"; exit 1; }
$HPY -m train.merge_gens --inputs $SAMPLES $BR --out $R/union_data
for run in fbr_only fbr_union; do
  G=$([ $run = fbr_only ] && echo $BR || echo $R/union_data)
  if [ ! -f $OUT/$run/model.safetensors ]; then
    wait_mem 10000; echo "=== [gpu$GPU] train $run $(date)"
    CUDA_VISIBLE_DEVICES=$GPU $HPY -m train.rft_lora --gens $G --split train --model $M --prefix "$COT" --label gold \
      --lr 1e-4 --epochs 2 --out $OUT/$run > $L/train_$run.log 2>&1
  fi
  grep "training solutions" $L/train_$run.log
  mkdir -p $R/$run; cp $OUT/$run/rft_config.json $R/$run/ 2>/dev/null || { echo "  TRAIN FAILED"; continue; }
  for t in 1 2 3; do
    [ -f $R/$run/fulltest/test.metrics.json ] && break
    wait_mem 6000
    CUDA_VISIBLE_DEVICES=$GPU $HPY -m methods.run --method greedy --model $OUT/$run --prefix "$COT" --split test \
      --max_new_tokens 320 --batch_problems 1 --out $R/$run/fulltest > $L/eval_${run}_fulltest.log 2>&1
  done
  echo "  $run FULL TEST $(acc $R/$run/fulltest/test.metrics.json)   (RFT-686 0.437, prompt-only 0.381)"
done
echo "=== done $(date)"
