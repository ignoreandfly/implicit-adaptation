# Roadmap

Tick a box when the task is merged. Put your name after a task when you pick it up.
Open decisions are listed at the bottom; don't build anything that depends on them yet.

## Direction (after mentor meeting)

Hewitt et al.'s rule-based adapter (a Python program that edits the next-token
distribution, no training) combined with inference-time scaling, evaluated on GSM8K first.

To separate what SFT and RL each contribute, we use **OLMo 2 1B**, which releases every
post-training stage. (Qwen2.5-0.5B was dropped from the baselines after the smoke test.)

| Family | Checkpoint | Post-training | Prompt |
|---|---|---|---|
| OLMo 2 1B | `allenai/OLMo-2-0425-1B` | none | plain `Question:/Answer:` |
| OLMo 2 1B | `allenai/OLMo-2-0425-1B-SFT` | SFT (Tülu 3 mix, includes math CoT) | chat template |
| OLMo 2 1B | `allenai/OLMo-2-0425-1B-DPO` | SFT + DPO | chat template |
| OLMo 2 1B | `allenai/OLMo-2-0425-1B-Instruct` | SFT + DPO + RLVR (math) | chat template |

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
- [x] Sampling backend for n > 1: `--backend vllm` (vLLM 0.9.2, env `implicit-vllm`, see README); the HF
      path drops to batch size 1 at n = 64. `--subset 500` = fixed 500-problem test subset (seed 0)
- [x] Colab notebook (`notebooks/colab_baselines.ipynb`): smoke test + full runs, results on Drive
- [x] Smoke test: 50 questions per model, check outputs by eye, record real run times (results below)
- [x] Look at truncated outputs: Base truncations are repetition loops; DPO/Instruct truncations are long
      self-correcting derivations ("Let's re-evaluate"). Keep 512 tokens (more would mostly add time) and report
      `truncation_rate`. Optional check: Instruct smoke with `--max_new_tokens 1024`.
- [ ] Phase 2 notes: Base loop rate is a baseline for the adapter's repetition rule; self-correction phrases
      (none in SFT, most in RLVR) are candidate forking tokens for the entropy/DTR analysis
- [ ] `format_rate` undercounts chat models (needs >= 2 `a op b = c` equations; misses `2 (blue) + 1 (white) = 3`, `\frac`)

### Runs (greedy pass@1, max 512 new tokens)
- [x] OLMo 2 1B Base: zero-shot and 8-shot
- [x] OLMo 2 1B SFT: zero-shot chat
- [x] OLMo 2 1B DPO: zero-shot chat
- [x] OLMo 2 1B Instruct (RLVR): zero-shot chat
- [ ] Sampling runs for pass@k and maj@k (size depends on the n = 64 decision below)

### Full test set (1,319 problems, greedy, RTX 3090, bf16)

Run with `scripts/gsm8k_baselines.sh`; metrics in `results/olmo2-1b-*/gsm8k/`.

| Run | strict | lenient | format | step acc | mean tokens | truncated | sec |
|---|---|---|---|---|---|---|---|
| OLMo Base 0-shot | 0.003 | 0.083 | 0.074 | 0.620 | 58 | 0.028 | 303 |
| OLMo Base 8-shot | 0.326 | 0.328 | 0.788 | 0.800 | 86 | 0.033 | 1077 |
| OLMo SFT | 0.582 | 0.594 | 0.732 | 0.818 | 149 | 0.014 | 547 |
| OLMo DPO | 0.553 | 0.600 | 0.648 | 0.862 | 167 | 0.014 | 830 |
| OLMo Instruct (RLVR) | 0.627 | 0.674 | 0.679 | 0.867 | 171 | 0.046 | 935 |

95% CI is about ±2.7 points at n = 1,319. Lenient accuracy matches the model card for DPO (60.0 vs 59.0)
and Instruct (67.4 vs 68.3); SFT is ~7 points above its card (59.4 vs 52.1), likely prompt/extraction
differences. SFT gives the main jump over Base, DPO adds step accuracy but not accuracy, RLVR adds ~7 points.
Times are with DPO, Instruct and 8-shot sharing one GPU, so they overstate single-run cost.

### Smoke test (first 50 test problems, greedy, T4)

| Run | strict | lenient | format | step acc | mean tokens | truncated | sec |
|---|---|---|---|---|---|---|---|
| OLMo Base 0-shot | 0.00 | 0.04 | 0.02 | 0.56 | 60 | 0.04 | 84 |
| OLMo Base 8-shot | 0.26 | 0.26 | 0.74 | 0.81 | 91 | 0.04 | 178 |
| OLMo SFT | 0.56 | 0.56 | 0.64 | 0.81 | 144 | 0.00 | 61 |
| OLMo DPO | 0.58 | 0.60 | 0.60 | 0.89 | 176 | 0.04 | 88 |
| OLMo Instruct (RLVR) | 0.56 | 0.60 | 0.54 | 0.89 | 197 | 0.10 | 93 |

n = 50, so differences under ~10 points are noise. Model-card GSM8K: SFT 52.1, DPO 59.0, Instruct 68.3.
Observations: Base answers zero-shot without any reasoning (the target for the Hewitt adapter);
SFT sometimes writes Python code instead of reasoning (Tülu 3 has code-based math data) and
must guess the program's output; Instruct sometimes copies the instruction ("The answer is N.").

### Analysis
- [ ] Results table: strict / lenient accuracy, format rate, step accuracy, length, truncation
- [ ] Token signals on saved outputs: entropy, share of high-entropy "fork" tokens
      (Wang et al. 2025), deep-thinking ratio (Chen et al. 2026); split by correct/incorrect and by stage
- [ ] Short write-up: what each stage (SFT, DPO, RL) adds, and whether the signals exist at this size

## Phase 2: methods (to be designed after Phase 1)

- [ ] Hewitt rule-based adapter (`adapters/`): EOS boost with length, fixed shift on a few tokens, repetition penalty
- [ ] Inference-time scaling ideas: fork-aware decoding (raise temperature / branch at high-entropy tokens),
      DTR-weighted answer selection, encouraging deep tokens during decoding

### vLLM backend check (RTX 3090, GPU 1 shared with other jobs at ~95% use, `--gpu_memory_utilization 0.35`)

Greedy, first 100 test problems; "same" = identical output text.

| Run | HF bs 64 (saved) lenient | HF bs 1 lenient | vLLM lenient | same: vLLM vs HF bs 64 | same: HF bs 1 vs HF bs 64 |
|---|---|---|---|---|---|
| OLMo Base 0-shot | 0.09 | 0.08 | 0.10 | 0.70 | 0.75 |
| OLMo SFT | 0.54 | 0.57 | 0.56 | 0.55 | 0.54 |

vLLM differs from HF about as much as HF at batch 64 differs from HF at batch 1 (bf16 + padding), so the
differences are numerical near-ties, not a prompt/stop bug (inputs are identical token ids).

Sampling n = 64, T = 0.8, first 50 problems of the 500 subset:

| Run | pass@1 | pass@16 | pass@64 | maj@16 | maj@64 | mean tokens | gen sec | wall sec |
|---|---|---|---|---|---|---|---|---|
| OLMo Base 0-shot | 0.063 | 0.480 | 0.720 | 0.12 | 0.16 | 65 | 71 | 104 |
| OLMo Base 8-shot | 0.213 | 0.719 | 0.860 | 0.42 | 0.46 | 79 | 95 | n/a |
| OLMo Instruct (RLVR) | 0.644 | 0.891 | 0.920 | 0.78 | 0.76 | 172 | 193 | 226 |

pass@k is lenient; wall includes ~30 s of vLLM startup. Estimate for 500 problems x n = 64: Base 0-shot
~12 min, Base 8-shot ~16 min, SFT / DPO / Instruct ~28-32 min each, so ~2-2.5 h for all 5 configs on a
shared GPU 1 at 0.35 memory (KV cache ~37k tokens; more memory = more concurrent sequences = faster).

## Rough compute (one L4, to be replaced by smoke-test timings)

| Job | Per config | All configs |
|---|---|---|
| Greedy pass@1 | ~5–15 min | ~1.5–2 h |
| Sampling n = 16 (vLLM) | ~10–20 min | ~2–3 h |
| Sampling n = 64 (vLLM) | ~40–60 min | ~6–8 h |
| Sampling n = 64, 500-problem subset (vLLM 0.9.2, shared RTX 3090, measured on 50) | ~12–32 min | ~2–2.5 h |
| Entropy / DTR pass on saved outputs | ~5–10 min | ~1 h |

## Open decisions

- [ ] **Sampling budget:** n = 64 on the full test set, n = 64 on a fixed 500-problem subset, or n = 16 only
- [ ] GPU and environment for the full runs: Colab for now; GCP L4 for sampling runs? (vLLM works on the lab 3090s, env `implicit-vllm`)
- [ ] 8-shot for the post-trained models too, or zero-shot chat only
