#!/usr/bin/env bash
# 1B method exploration for a dedicated GPU (RUNBOOK.md section 7). Goal: a training-light method that moves
# OLMo 2 1B SFT from ~60% toward the RL model (Instruct ~68%) on GSM8K test500 at 1x inference cost.
#   SMOKE=1 bash scripts/gcp_1b_explore.sh        # tiny end-to-end check of every step (~20 min) - run this first
#   bash scripts/gcp_1b_explore.sh                # full run (~8-12 GPU-hours on one A100; steps are resumable)
#   STEPS="data rft decision forkbranch eval" ... # subset of steps
# Env: VPY = python of the vllm env, HPY = python of the hf/peft env (RUNBOOK section 1).
set -uo pipefail
cd "$(dirname "$0")/.."
VPY=${VPY:-python}; HPY=${HPY:-python}; MEM=${MEM:-0.85}
STEPS=${STEPS:-"data rft decision forkbranch eval"}
SMOKE=${SMOKE:-0}
if [ "$SMOKE" = 1 ]; then NQ=40; NS=8; R=results/explore_smoke; else NQ=3000; NS=16; R=results/explore; fi
CK=${CK:-checkpoints/explore$([ "$SMOKE" = 1 ] && echo _smoke)}
mkdir -p $R $CK logs
SAMPLES=$R/sft_train${NQ}_n${NS}           # SFT samples on training questions (ReasonMaxxer-style rollouts)
BRANCH=$R/fork_branch_sft_train${NQ}       # greedy + fork branches on the same questions
NR=$([ "$SMOKE" = 1 ] && echo 40 || echo 300)
RMX=$R/sft_train${NR}_n20_T0.6             # ReasonMaxxer recipe data: 300 problems x 20 rollouts at T 0.6
has () { [[ " $STEPS " == *" $1 "* ]]; }
acc () { grep -oE "'acc_lenient': [0-9.]+" $1 | tail -1; }

[ -f data/gsm8k/test500.jsonl ] || { echo "prepare data first (RUNBOOK section 1)"; exit 1; }

if has data; then
  [ -f $SAMPLES/train.generations.jsonl ] || $VPY -m evaluation.generate --backend vllm --model olmo2-1b-sft --split train \
     --limit $NQ --n_samples $NS --temperature 0.7 --gpu_memory_utilization $MEM --out $SAMPLES > logs/explore_samples.log 2>&1
  [ -f $BRANCH/train.generations.jsonl ] || $VPY -m train.fork_branch_data --model olmo2-1b-sft --n $NQ --forks 4 --m 2 \
     --gpu_memory_utilization $MEM --out $BRANCH > logs/explore_branch.log 2>&1
  [ -f $RMX/train.generations.jsonl ] || $VPY -m evaluation.generate --backend vllm --model olmo2-1b-sft --split train \
     --limit $NR --n_samples 20 --temperature 0.6 --gpu_memory_utilization $MEM --out $RMX > logs/explore_rmx.log 2>&1
  echo "data: samples $(cat $SAMPLES/train.metrics.json 2>/dev/null | grep -oE '"acc_lenient": [0-9.]+' | head -1) | $(tail -1 logs/explore_branch.log)"
fi

train () {  # tag, trainer module, args...
  tag=$1; mod=$2; shift 2
  [ -f $CK/$tag/model.safetensors ] || $HPY -m $mod "$@" --out $CK/$tag > logs/explore_train_$tag.log 2>&1
  [ -f $CK/$tag/model.safetensors ] && echo "trained $tag" || echo "TRAINING FAILED $tag (logs/explore_train_$tag.log)"
}
LIM=$([ "$SMOKE" = 1 ] && echo "--limit $NQ" || echo "")
if has rft; then        # baseline (positive-only = ReasonMaxxer's ablation), on the same data
  train rft_gold     train.rft_lora --gens $SAMPLES --model olmo2-1b-sft --label gold     --lr 1e-4 --epochs 2
  train rft_majority train.rft_lora --gens $SAMPLES --model olmo2-1b-sft --label majority --lr 1e-4 --epochs 2
fi
if has decision; then   # direction 1: faithful ReasonMaxxer (reference-code recipe: 50 mid-difficulty problems, tau 1.4
                        # nats, lr 1e-4, <=149 steps, clarity curriculum, KL 0.2) + tau sweep + positive-only ablation
  for tau in 1.4 1.0 2.2; do
    train rmx_gold_tau$tau     train.decision_lora --gens $RMX --model olmo2-1b-sft --label gold     --tau $tau
  done
  train rmx_majority_tau1.4    train.decision_lora --gens $RMX --model olmo2-1b-sft --label majority --tau 1.4
  train rmx_gold_posonly       train.decision_lora --gens $RMX --model olmo2-1b-sft --label gold     --tau 1.4 --positive_only
  train rmx_gold_allprobs      train.decision_lora --gens $SAMPLES --model olmo2-1b-sft --label gold --tau 1.4 --n_problems 0
fi
if has forkbranch; then # direction 2: fork-branch self-training (ours)
  train fb_rft_gold      train.rft_lora      --gens $BRANCH --model olmo2-1b-sft --label gold     --lr 1e-4 --epochs 2
  train fb_rft_majority  train.rft_lora      --gens $BRANCH --model olmo2-1b-sft --label majority --lr 1e-4 --epochs 2
  train fb_dec_gold      train.decision_lora --gens $BRANCH --model olmo2-1b-sft --label gold     --n_problems 0
  train fb_dec_majority  train.decision_lora --gens $BRANCH --model olmo2-1b-sft --label majority --n_problems 0
fi
if has eval; then       # greedy test500 (+ dev) through ONE pipeline (vLLM), incl. the SFT and Instruct references
  SPLIT=$([ "$SMOKE" = 1 ] && echo "--split dev --limit 40" || echo "--subset 500")
  for m in olmo2-1b-sft olmo2-1b-instruct $(ls -d $CK/*/ 2>/dev/null); do
    m=${m%/}; tag=$(basename $m)
    [ -f $R/eval/$tag/test.metrics.json ] || [ -f $R/eval/$tag/dev.metrics.json ] || \
      $VPY -m evaluation.generate --backend vllm --model $m --format chat $SPLIT --gpu_memory_utilization $MEM \
        --out $R/eval/$tag > logs/explore_eval_$tag.log 2>&1
    echo "eval $tag: $(acc logs/explore_eval_$tag.log)"
  done
fi
echo "=== gcp_1b_explore done $(date)"
