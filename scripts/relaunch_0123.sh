#!/usr/bin/env bash
# Relaunch (2026-11-23 ~01:15 ET): fork-LoRA round 2 + sanity check on GPU 0, 7B SFT n=64 on GPU 2.
# Waits use explicit PIDs (pattern-based waits matched their own launcher shells).
cd "$(dirname "$0")/.."
kill 3636323 2>/dev/null   # old sanity waiter (pattern wait would never end)

# round 2 without the stale wait
sed -i '/^while pgrep -f "scripts\/fork_lora_1b.sh"/d' scripts/fork_lora_1b_v2.sh
setsid nohup bash scripts/fork_lora_1b_v2.sh > logs/fork_lora_1b_v2.log 2>&1 < /dev/null &
V2=$!

# sanity check waits for round 2 by PID
sed -i 's/^while pgrep -f "scripts\/fork_lora_1b_v2.sh" >\/dev\/null; do sleep 30; done$/while kill -0 "$WAIT_PID" 2>\/dev\/null; do sleep 30; done/' scripts/fork_lora_sanity.sh
WAIT_PID=$V2 setsid nohup bash scripts/fork_lora_sanity.sh > logs/fork_lora_sanity.log 2>&1 < /dev/null &

# 7B SFT n=64 (first attempt was killed when the previous session ended)
cat > scripts/olmo7b_sft_n64.sh <<'EOS'
#!/usr/bin/env bash
cd "$(dirname "$0")/.."
export HF_HUB_CACHE=/data/azfarm/siddhant_hf_cache CUDA_VISIBLE_DEVICES=${GPU:-2}
$HOME/miniconda3/envs/implicit-vllm/bin/python -m evaluation.generate --backend vllm --model olmo2-7b-sft --subset 500 \
  --n_samples 64 --temperature 0.7 --gpu_memory_utilization 0.75 --out results/olmo2-7b-sft/gsm8k/0shot_sub500_n64_t0.7
EOS
chmod +x scripts/olmo7b_sft_n64.sh
GPU=2 setsid nohup bash scripts/olmo7b_sft_n64.sh > logs/olmo7b_sft_n64.log 2>&1 < /dev/null &
echo "launched round2=$V2"
