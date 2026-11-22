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
