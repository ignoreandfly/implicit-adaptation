#!/usr/bin/env bash
# Fork-contrastive self-distillation on OLMo 2 1B SFT (GPU 0, overnight 2026-11-22/23).
# Waits for the 1B FOBAR k=16 job, then: fork-pair data on 1,000 GSM8K training questions ->
# LoRA variants (majority labels x 2 lrs, gold labels, shuffled-label control) -> eval each (dev greedy,
# test500 greedy, test500 n=16) next to the SFT baseline through the same vLLM pipeline -> summary + push.
set -uo pipefail
cd "$(dirname "$0")/.."
until [ -f results/fobar/olmo2-1b-sft_sampling_k16.json ] && ! pgrep -f "analysis.fobar --pool sampling" >/dev/null; do sleep 60; done
export CUDA_VISIBLE_DEVICES=${GPU:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VPY=$HOME/miniconda3/envs/implicit-vllm/bin/python
HPY=$HOME/miniconda3/envs/ICCV/bin/python
PAIRS=data/fork_pairs/olmo2-1b-sft_train1000.jsonl
OUT=/data/azfarm/siddhant_fork_lora
R=results/fork_lora
mkdir -p data/fork_pairs $OUT $R

if [ ! -f $PAIRS ]; then
  echo "=== fork pairs $(date)"
  $VPY -m train.fork_pairs --model olmo2-1b-sft --n 1000 --gpu_memory_utilization 0.5 --out $PAIRS 2>&1 | grep -vE "INFO|WARN|it/s" | tail -3
fi

evalm () {  # $1 = model (registry name or path), $2 = tag
  for spec in "dev:" "test500:"; do
    split=${spec%:*}
    $VPY -m evaluation.generate --backend vllm --model $1 --format chat --split $split --gpu_memory_utilization 0.5 \
      --out $R/$2/greedy_$split 2>&1 | grep -E "acc_lenient" | tail -1 | sed "s/^/  $2 $split greedy: /"
  done
  $VPY -m evaluation.generate --backend vllm --model $1 --format chat --split test500 --n_samples 16 --temperature 0.7 \
    --gpu_memory_utilization 0.5 --out $R/$2/n16_test500 2>&1 | grep -E "acc_lenient" | tail -1 | sed "s/^/  $2 test500 n16: /"
}

echo "=== baseline SFT $(date)"; evalm olmo2-1b-sft sft_baseline
train () {  # $1 = tag, rest = args
  tag=$1; shift
  echo "=== train $tag $(date)"
  $HPY -m train.fork_lora --pairs $PAIRS --model olmo2-1b-sft "$@" --out $OUT/$tag 2>&1 | grep -E "training forks|saved|Error" | tail -3
  [ -f $OUT/$tag/model.safetensors ] && { echo "=== eval $tag $(date)"; evalm $OUT/$tag $tag; cp $OUT/$tag/fork_lora_config.json $R/$tag/ 2>/dev/null; }
}
train major_lr1e-4  --label majority --lr 1e-4
train gold_lr1e-4   --label gold --lr 1e-4
train shuffled_lr1e-4 --label majority --lr 1e-4 --shuffle
train major_lr3e-4  --label majority --lr 3e-4

echo "=== summary $(date)"
$HPY - <<'EOF' > results/fork_lora_summary.md
import json, os
R = "results/fork_lora"
def g(p, k):
    try: return json.load(open(p))[k]
    except Exception: return None
print("# Fork-contrastive self-distillation, OLMo 2 1B SFT (GSM8K)\n")
print("| Variant | dev greedy | test500 greedy | test500 maj@16 | test500 pass@16 | train forks |")
print("|---|---|---|---|---|---|")
for tag in ["sft_baseline", "major_lr1e-4", "major_lr3e-4", "gold_lr1e-4", "shuffled_lr1e-4"]:
    d = g(f"{R}/{tag}/greedy_dev/dev.metrics.json", "acc_lenient")
    t = g(f"{R}/{tag}/greedy_test500/test500.metrics.json", "acc_lenient")
    m = g(f"{R}/{tag}/n16_test500/test500.metrics.json", "maj@16")
    p = g(f"{R}/{tag}/n16_test500/test500.metrics.json", "pass@16_lenient")
    n = g(f"{R}/{tag}/fork_lora_config.json", "train_forks")
    f = lambda x: f"{x:.3f}" if isinstance(x, float) else "-"
    print(f"| {tag} | {f(d)} | {f(t)} | {f(m)} | {f(p)} | {n if n is not None else '-'} |")
print("\nReference (test500): SFT greedy 0.604, Instruct greedy 0.678/0.682, SFT maj@16 0.736.")
EOF
cat results/fork_lora_summary.md
git add results/fork_lora_summary.md results/fork_lora/*/*/*.metrics.json results/fork_lora/*/fork_lora_config.json \
        train/fork_pairs.py train/fork_lora.py train/__init__.py scripts/fork_lora_1b.sh 2>/dev/null
git commit -q -m "Fork-contrastive self-distillation on 1B SFT: data, LoRA variants, eval (overnight)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q 2>&1 | tail -1
echo "=== done $(date)"
