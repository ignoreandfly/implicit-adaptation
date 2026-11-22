# Replicate runs on the seahorse server (8x A6000), 2026-11-22

Same scripts and settings as the main runs, re-run on a second machine (uv envs: torch 2.7.1 / transformers 4.57.1 /
peft 0.17.0 / datasets 3.6.0; vllm 0.9.2). The main-run files at the same paths under `results/` are the ones the
README reports; these are kept as a replicate.

| Run | main (README) | seahorse replicate |
|---|---|---|
| Base + CoT RFT gold, test500 / dev | 45.6% / 51.5% (2,183 training solutions) | 42.6% / 49.0% (2,162) |
| Base + CoT RFT majority, test500 / dev | 41.4% / 47.0% | 39.6% / 47.0% |

The difference comes from the n = 8 training samples (HF sampling, different machine/batching), not the settings.

Note: `results/diversity/` and `results/rft_base/base_rft_gold_forkdpo_gold/` (queue_base_forkvsrft.sh) were run on
seahorse and use the seahorse RFT gold model (42.6%), and a seahorse rebuild of Base fork-LoRA gold.

## Last experiments on seahorse (2026-11-22, `queue_base_forkvsrft.sh`, `queue_sft_rft.sh`)

### Base + zero-shot CoT, same 1,000 training questions (test500, lenient)

| Model | greedy | pass@1 (T 0.7) | maj@16 | pass@16 |
|---|---|---|---|---|
| Base + CoT | 37.0% (main) | 24.8% | 46.8% | 73.6% |
| fork-LoRA gold (seahorse rebuild; reproduces 40.2 / 39.2) | 41.2% | 31.1% | 50.8% | 76.8% |
| RFT gold (seahorse) | 42.6% | **37.6%** | **54.8%** | **78.8%** |
| RFT gold + fork-DPO (gold, on-policy pairs from the RFT model) | 38.8% (dev 42.0 vs RFT 49.0) | 32.9% | 52.4% | 76.2% |

Files: `results/diversity/*/test500.metrics.json` (n = 16, `methods.run --method sample`, 320 tokens),
`results/fork_lora_base/base_dpo_gold/rebuild_test500/`, `results/rft_base/base_rft_gold_forkdpo_gold/`.

- RFT does not collapse diversity: it beats fork-LoRA on every column, pass@16 included.
- Fork-DPO on top of RFT hurts (-3.8 test500, -7 dev): fork training does not add to RFT.

### RFT on 1B SFT, all 7,273 training questions, 1 epoch (dev, 200)

Compare each model with SFT **in its own decoder** (greedy at 1B moves with the code path):

| Model | decoder | dev | SFT in that decoder | delta |
|---|---|---|---|---|
| fork-LoRA full-data gold (main) | HF `rl_rules`, batch 2 | 71.0% | 72.0% | -1.0 |
| fork-LoRA full-data shuffled (main) | HF `rl_rules`, batch 2 | 62.5% | 72.0% | -9.5 |
| RFT 1,500 q, 2 epochs, gold / majority (main) | HF `rl_rules`, batch 2 | 75.5% / 74.5% | 72.0% | +3.5 / +2.5 |
| RFT 7,273 q, 1 epoch, gold (seahorse) | vLLM greedy, chat | 72.0% | 69.5% | +2.5 |
| RFT 7,273 q, 1 epoch, majority / label-free (seahorse; majority correct on 5,877 / 7,273) | vLLM greedy, chat | 71.5% | 69.5% | +2.0 |

Files: `results/rft_sft/sft_rft_full7273_{gold,majority}/`. At n = 200 the standard error is about 3 points, so the
SFT deltas are within noise, but they all point the same way: RFT >= SFT >= fork-LoRA. Label-free RFT on SFT is
close to gold RFT, since SFT's majority is right on 81% of training questions.
