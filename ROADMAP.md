# Roadmap

Tick a box when the task is merged. Put your name after a task when you pick it up.
Open decisions are listed at the bottom; don't build anything that depends on them yet.

## Direction (after mentor meeting)

Hewitt et al.'s rule-based adapter (a Python program that edits the next-token
distribution, no training) combined with inference-time scaling, evaluated on GSM8K first.

To separate what SFT and RL each contribute, the main model family is **OLMo 2 1B**, which
releases every post-training stage. **Qwen2.5-0.5B** (Base vs. Instruct) is the second family.

| Family | Checkpoint | Post-training | Prompt |
|---|---|---|---|
| OLMo 2 1B | `allenai/OLMo-2-0425-1B` | none | plain `Question:/Answer:` |
| OLMo 2 1B | `allenai/OLMo-2-0425-1B-SFT` | SFT (Tülu 3 mix, includes math CoT) | chat template |
| OLMo 2 1B | `allenai/OLMo-2-0425-1B-DPO` | SFT + DPO | chat template |
| OLMo 2 1B | `allenai/OLMo-2-0425-1B-Instruct` | SFT + DPO + RLVR (math) | chat template |
| Qwen2.5 | `Qwen/Qwen2.5-0.5B` | none | plain `Question:/Answer:` |
| Qwen2.5 | `Qwen/Qwen2.5-0.5B-Instruct` | SFT + DPO + GRPO (learned reward model) | chat template |

## Phase 1: baseline numbers on GSM8K (no new methods)

Goal: report what each existing checkpoint does on the GSM8K test set (1,319 problems).

### Setup
- [x] GSM8K loader and verifier (`tasks/gsm8k.py`): drops `<<calc>>` notes, ends solutions with
      "Therefore, the answer is N.", normalizes `$`, commas and decimals
- [x] Model registry (`common/models.py`): short name, HF id, prompt format (plain vs. chat template)
- [x] `evaluation/generate.py`: chat template for post-trained models (few-shot = earlier turns), bf16/float32 by GPU
- [x] Stop strings per prompt format: chat models stop only at EOS (`\n\n` cuts off multi-paragraph answers)
- [x] Verifier: accept `\boxed{N}`, `#### N`, markdown bold and LaTeX (`$`, `\times`) in answers and equations
- [x] Log generated length and the share of outputs that hit `max_new_tokens`
- [x] Add maj@k (majority vote) to `evaluation/metrics.py`
- [ ] Sampling backend for n > 1 (vLLM); the HF path drops to batch size 1 at n = 64
- [x] Colab notebook (`notebooks/colab_baselines.ipynb`): smoke test + full runs, results on Drive
- [ ] Smoke test: 50 questions per model, check outputs by eye, record real run times

### Runs (greedy pass@1, max 512 new tokens)
- [ ] OLMo 2 1B Base: zero-shot and 8-shot
- [ ] OLMo 2 1B SFT: zero-shot chat
- [ ] OLMo 2 1B DPO: zero-shot chat
- [ ] OLMo 2 1B Instruct (RLVR): zero-shot chat
- [ ] Qwen2.5-0.5B Base: zero-shot and 8-shot
- [ ] Qwen2.5-0.5B Instruct: zero-shot chat
- [ ] Sampling runs for pass@k and maj@k (size depends on the n = 64 decision below)

### Analysis
- [ ] Results table: strict / lenient accuracy, format rate, step accuracy, length, truncation
- [ ] Token signals on saved outputs: entropy, share of high-entropy "fork" tokens
      (Wang et al. 2025), deep-thinking ratio (Chen et al. 2026); split by correct/incorrect and by stage
- [ ] Short write-up: what each stage (SFT, DPO, RL) adds, and whether the signals exist at this size

## Phase 2: methods (to be designed after Phase 1)

- [ ] Hewitt rule-based adapter (`adapters/`): EOS boost with length, fixed shift on a few tokens, repetition penalty
- [ ] Inference-time scaling ideas: fork-aware decoding (raise temperature / branch at high-entropy tokens),
      DTR-weighted answer selection, encouraging deep tokens during decoding

## Rough compute (one L4, to be replaced by smoke-test timings)

| Job | Per config | All configs |
|---|---|---|
| Greedy pass@1 | ~5–15 min | ~1.5–2 h |
| Sampling n = 16 (vLLM) | ~10–20 min | ~2–3 h |
| Sampling n = 64 (vLLM) | ~40–60 min | ~6–8 h |
| Entropy / DTR pass on saved outputs | ~5–10 min | ~1 h |

## Open decisions

- [ ] **Sampling budget:** n = 64 on the full test set, n = 64 on a fixed 500-problem subset, or n = 16 only
- [ ] GPU and environment for the full runs: Colab for now; GCP L4 for sampling runs? vLLM installable?
- [ ] 8-shot for the post-trained models too, or zero-shot chat only
