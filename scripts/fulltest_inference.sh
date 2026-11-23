#!/usr/bin/env bash
# Inference-time methods on the FULL GSM8K test set (1,319) for OLMo 2 1B Base and SFT, plus Instruct greedy as the
# RL reference in the same harness. ONE GPU ($GPU); five lanes share it. Settings = each method's best on Base dev
# (results/screen/olmo2-1b-base/dev); Base keeps the dev protocol (256 new tokens, same batch sizes), SFT/Instruct
# use chat format with 512 new tokens and the same settings (not re-tuned on SFT).
# Output: results/fulltest/<model>/<name>/test.{metrics.json,generations.jsonl}; resumable (finished runs skipped).
#   GPU=0 HPY=~/envs/ia-hf/bin/python bash scripts/fulltest_inference.sh
set -uo pipefail
cd "$(dirname "$0")/.."
HPY=${HPY:?}; GPU=${GPU:?}
export CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
R=results/fulltest; L=logs/fulltest; mkdir -p $R $L
COT=" Let's think step by step."
B=olmo2-1b-base; S=olmo2-1b-sft

job () {  # model name args...  (retries once after an OOM / crash)
  local model=$1 name=$2; shift 2
  local out=$R/$model/$name
  [ -f $out/test.metrics.json ] && { echo "skip $model/$name"; return; }
  local tok=256; [ $model != $B ] && tok=512
  for try in 1 2; do
    echo "=== $model/$name (try $try) $(date +%T)"
    $HPY -m methods.run --model $model --split test --max_new_tokens $tok --out $out "$@" > $L/${model}_$name.log 2>&1 && break
    tail -2 $L/${model}_$name.log; sleep 300
  done
  python3 - $out <<'EOF'
import json, sys
try:
    m = json.load(open(sys.argv[1] + "/test.metrics.json"))
    print("   ", {k: round(v, 3) for k, v in m.items() if k.startswith("acc") and isinstance(v, float)}, "sec", m.get("seconds"))
except Exception as e:
    print("    FAILED", e)
EOF
}
rescore () {  # model name
  local out=$R/$1/$2
  [ -f $out/test.selection.json ] && return
  [ -f $out/test.metrics.json ] || return
  echo "=== rescore $1/$2 $(date +%T)"
  $HPY -m methods.rescore $out --split test > $L/${1}_$2_rescore.log 2>&1
  grep -E '^(single|majority|oracle|eba|borda1.0:self_certainty|dc90:|top50:|wvote:) ' $L/${1}_$2_rescore.log | head -12 | sed 's/^/    /'
}

lane1 () {
  job $S tlsc_K4 --method tlsc --tlsc_k 4 --tlsc_steps 40 --temperature 0.7 --batch_problems 4
  job $B greedy_0shot --method greedy --batch_problems 8
  job $B greedy_cot --method greedy --prefix "$COT" --batch_problems 8
  job $B forced_lets --method greedy --prefix " Let's" --batch_problems 8
  job $B forced_first --method greedy --prefix " First," --batch_problems 8
  job olmo2-1b-instruct greedy --method greedy --batch_problems 8
  job $S greedy --method greedy --batch_problems 8
}
lane2 () {
  job $B tlsc_K4 --method tlsc --prefix "$COT" --tlsc_k 4 --tlsc_steps 40 --temperature 0.7 --batch_problems 4
  job $B power_smc_N16 --method power_smc --prefix "$COT" --smc_alpha 4.0 --n 16 --smc_ramp 100 --batch_problems 4
  job $B dola_low_rep1.2 --method dola --dola_layers low --rep_penalty 1.2 --batch_problems 8
  job $B budget_forcing_r1 --method budget_forcing --prefix "$COT" --rounds 1 --batch_problems 8
  job $B cfg_g1.25 --method cfg --prefix "$COT" --cfg_gamma 1.25 --batch_problems 8
}
lane3 () {
  job $S sc_n10 --method sample --n 10 --temperature 0.7 --batch_problems 8
  rescore $S sc_n10
  job $S eager_theta1.0 --method eager --eager_theta 1.0 --n 10 --batch_problems 4
  job $S power_smc_N16 --method power_smc --smc_alpha 4.0 --n 16 --smc_ramp 100 --batch_problems 4
}
lane4 () {
  job $B sc_n10 --method sample --prefix "$COT" --n 10 --temperature 0.7 --batch_problems 8
  rescore $B sc_n10
  job $B eager_theta1.0 --method eager --prefix "$COT" --eager_theta 1.0 --n 10 --batch_problems 4
  job $B cot_decoding_k10 --method cot_decoding --k 10 --batch_problems 8
  job $B rule_adapter_A --method rule_adapter --ra_stop_pen 3.0 --ra_first_digit_pen 10.0 --ra_first_step_bonus 6.0 \
    --ra_off rep --ra_n 1 --batch_problems 8
  job $B rule_adapter_A_vote --method rule_adapter --ra_stop_pen 3.0 --ra_first_digit_pen 10.0 --ra_first_step_bonus 6.0 \
    --ra_off rep --ra_n 10 --batch_problems 4
}
lane5 () {
  job $S cot_decoding_k10 --method cot_decoding --k 10 --batch_problems 8
  job $S dola_low_rep1.2 --method dola --dola_layers low --rep_penalty 1.2 --batch_problems 8
  job $S budget_forcing_r1 --method budget_forcing --rounds 1 --batch_problems 8
  job $S cfg_g1.25 --method cfg --cfg_gamma 1.25 --batch_problems 8
  job $S contrastive_b0.25 --method contrastive --cd_beta 0.25 --dropout 0.2 --batch_problems 8
  job $B contrastive_b0.25 --method contrastive --prefix "$COT" --cd_beta 0.25 --dropout 0.2 --batch_problems 8
}

for i in 1 2 3 4 5; do lane$i > $L/lane$i.out 2>&1 & sleep 20; done
wait
cat $L/lane*.out
git add $R scripts/fulltest_inference.sh methods/ 2>/dev/null
git commit -q -m "Inference-time methods on the full GSM8K test: 1B Base and SFT (+ Instruct greedy), chat support in methods

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
echo "=== all done $(date)"
