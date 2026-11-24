# Implicit Adaptation below 3B Parameters

COMS 4705 final project. Hewitt et al. ("Instruction Following without Instruction Tuning")
showed that a small hand-written program that edits the next-token distribution, with no
training, can make a base model follow instructions. We ask whether that kind of
rule-based adapter, combined with inference-time scaling, can bring out step-by-step
reasoning in models of 1B parameters or fewer, and how close it gets to models that went
through SFT and RL.

We evaluate on GSM8K first. To continue experiments on another GPU or machine, see [RUNBOOK.md](RUNBOOK.md). Task list and open decisions: [ROADMAP.md](ROADMAP.md).

## Models

We use OLMo 2 1B because it releases every post-training stage, so we can see what SFT,
DPO and RL each add, and its pretraining data is open.

| Name (`--model`) | Checkpoint | Post-training | Prompt |
|---|---|---|---|
| `olmo2-1b-base` | `allenai/OLMo-2-0425-1B` | none | plain |
| `olmo2-1b-sft` | `allenai/OLMo-2-0425-1B-SFT` | SFT (Tülu 3 mix, includes math CoT) | chat |
| `olmo2-1b-dpo` | `allenai/OLMo-2-0425-1B-DPO` | SFT + DPO | chat |
| `olmo2-1b-instruct` | `allenai/OLMo-2-0425-1B-Instruct` | SFT + DPO + RLVR (math) | chat |

The names are defined in `common/models.py`.

## Layout

```
common/      model registry, prompt construction (plain / chat), jsonl I/O
tasks/       gsm8k.py: loader (writes data/gsm8k/*.jsonl) + answer verifier
evaluation/  generate.py (HF or vLLM generation + scoring), metrics.py (accuracy, pass@k, maj@k, format, steps, length)
notebooks/   colab_baselines.ipynb: all baseline runs on Colab, results saved to Drive
adapters/    rule-based logits processors (TODO: Hewitt adapter)
analysis/    summarize.py -> results/summary.csv
data/        downloaded splits (gitignored)
results/     generations + metrics per run
```

## Inference-time methods

We screen training-free methods on OLMo 2 1B Base (zero-shot, GSM8K dev split, 200 problems)
with `methods/run.py`, then move the ones that beat the baselines to the test set and the other
stages. Most of these were shown on 7B–120B models; testing them at 1B is the point.
Status: **done** = results in `results/screen/`, **running** = current batch, **next** = queued after it.

### Current batch

| Method | Paper | Shown on | Status |
|---|---|---|---|
| CoT-decoding (branch on top-k first tokens, pick by answer confidence) | [Wang & Zhou 2024](https://arxiv.org/abs/2402.10200) | PaLM-2, Mistral-7B, Gemma-7B | done |
| Zero-shot CoT ("Let's think step by step.") and forced first token | [Kojima et al. 2022](https://arxiv.org/abs/2205.11916) | GPT-3-scale | done |
| DoLa (contrast final vs. early layer) | [Chuang et al. 2024](https://arxiv.org/abs/2309.03883); may hurt reasoning per [Shi et al. 2024](https://arxiv.org/abs/2402.06925) | LLaMA 7B–65B | done (hurts at 1B) |
| Budget forcing ("Wait" when the model stops) | [Muennighoff et al. 2025 (s1)](https://arxiv.org/abs/2501.19393) | Qwen-32B | done (hurts at 1B) |
| Self-consistency (majority vote over samples) | [Wang et al. 2023](https://arxiv.org/abs/2203.11171) | large models | running |
| Think@n (vote among the top half of samples by deep-thinking ratio) | [Chen et al. 2026](https://arxiv.org/abs/2602.13517) | 20B–120B | done (hurts at 1B) |
| Fork-aware temperature sampling (our adaptation of the fork-token finding) | idea from [Wang et al. 2025](https://arxiv.org/abs/2506.01939) | — | running; to be replaced by EAGer |

### Next (from the literature)

| # | Method | Paper | Shown on | Cost |
|---|---|---|---|---|
| 1 | Self-certainty best-of-N (Borda-weighted vote) | [Kang et al., NeurIPS 2025](https://arxiv.org/abs/2502.18581) | reasoning LLMs, with CoT | done: 54.0% (best so far) |
| 2 | DeepConf offline (filter by local/lowest-window confidence, then vote) | [Fu et al., ICLR 2026](https://arxiv.org/abs/2508.15260) | Qwen3, GPT-OSS | done: 52.0–52.5% |
| 3 | Classifier-free guidance on the prompt | [Sanchez et al., ICML 2024](https://arxiv.org/abs/2306.17806) | Pythia, GPT-2, LLaMA (incl. small); GSM8K | done (hurts at 1B) |
| 4 | Distillation contrastive decoding (dropout amateur, single model); original contrastive decoding reported to hurt small models | [Phan et al. 2024](https://arxiv.org/abs/2402.14874); [O'Brien & Lewis 2023](https://arxiv.org/abs/2309.09117) | LLaMA; GSM8K | done (hurts at 1B) |
| 5 | EM-INF (per-step logit adjustment that lowers entropy) | [Agarwal et al., NeurIPS 2025](https://arxiv.org/abs/2505.15134) | Qwen 7B/32B | low–medium |
| 6 | EAGer / entropy-gated branching (branch only at high-entropy tokens) | [EAGer 2025](https://arxiv.org/abs/2510.11170); [Entropy-gated branching](https://arxiv.org/abs/2503.21961) | small LLMs, +4.6% over greedy | medium |
| 7 | Thought-level self-certainty maximization (pick the most certain continuation per step) | [arXiv 2603.07159](https://arxiv.org/abs/2603.07159) | GSM8K/MATH500; up to 4× greedy on Qwen-1.5B | medium |
| 8 | Power sampling (MCMC on the sharpened base distribution; base matches GRPO) | [Karan & Du, ICLR 2026](https://arxiv.org/abs/2510.14901); cheaper: [token-level](https://arxiv.org/abs/2601.21590), [entropy-guided](https://arxiv.org/abs/2606.09926) | Qwen2.5 base models; MATH500 | high; Power-SMC variant done (below self-consistency at 1B) |
| 9 | CoT steering vector (add a "think step by step" direction to the residual stream) | [Zhang & Viteri 2024](https://arxiv.org/abs/2409.14026); SAE-free: [2505.15634](https://arxiv.org/abs/2505.15634) | Llama3-8B, Mistral-7B | low–medium |
| 10 | φ-Decoding (foresight sampling) | [Xu et al., ACL 2025](https://arxiv.org/abs/2503.13288) | LLaMA3.1-8B, +14% over CoT | high |
| 11 | Power-SMC (particle-filter power sampling in one batched decode; 1.4–3.3× latency vs 16–28× for MCMC) | [Azizi et al. 2026](https://arxiv.org/abs/2602.10273) | MATH500, three models | done: 37–40.5% (no gain over zero-shot CoT) |
| 12 | Depth-entropy guided power sampling (favor late layer-wise entropy collapse) | [Meng et al. 2026](https://arxiv.org/abs/2607.09693) | three models; ≈ GRPO in-domain | high |
| 13 | Better aggregation than majority vote: prefix-consistency weighted vote (regenerate from truncated prefixes) | [Iwase et al. 2026](https://arxiv.org/abs/2605.07654) | five reasoning models; MV accuracy at median 4.6× fewer tokens | medium |
| 14 | Better aggregation: embedding-based agreement (cluster samples in hidden-state space) | [Ontalvilla et al. 2026](https://arxiv.org/abs/2606.12003) | math, code, summarization | rescoring only |
| 15 | Better aggregation: ARBITER (fix wrong-majority basins with same-model evidence / hidden states) | [Cai et al. 2026](https://arxiv.org/abs/2605.26172) | Qwen3-4B, GSM8K | rescoring only |
| 16 | Better aggregation: delegation voting (PPV; entropy confidence + embedding similarity) | [Sakai et al. 2026](https://arxiv.org/abs/2606.08098) | MMLU-Pro | rescoring only |
| 17 | REWIND (entropy-gradient backtracking + MH-style filtering of regenerations) | [arXiv 2610.19561](https://arxiv.org/abs/2610.19561) | three model families | medium–high |
| 18 | DenseSteer (steering vector from 50 own solutions rewritten "dense" by an external LLM) — candidate for the cheap replication stage | [Ouyang et al., ICML 2026](https://arxiv.org/abs/2605.29247) | Qwen2.5 / Llama 1B–8B Instruct | ≈ greedy |
| 19 | USteer (steer activations along the gradient of a confidence measure) — candidate for the cheap replication stage | [NeurIPS 2026](https://arxiv.org/abs/2609.38962) | reasoning / hallucination tasks | ≈ greedy |
| — | Hewitt et al. rule-based adapter | [Hewitt et al. 2024](https://arxiv.org/abs/2409.14254) | Llama-2 7B, OLMo 7B | after the above |

### Screening results so far (OLMo 2 1B Base, GSM8K dev, 200 problems, lenient accuracy)

| Method | Acc. | Tokens / problem |
|---|---|---|
| Greedy zero-shot (baseline) | 9.5% | 40 |
| Greedy 8-shot (prompting reference) | 38.5% | 73 |
| **Self-consistency on zero-shot CoT** (n = 10, T = 0.7, majority vote) | **51.0%** | 1308 |
| Self-consistency + self-certainty Borda vote (Kang et al.) on the same 10 samples | 54.0% | 1308 |
| Self-consistency + DeepConf filter (keep top 90%, confidence-weighted vote) | 52.0–52.5% | 1308 |
| Self-consistency + Think@n (top half by deep-thinking ratio, then vote) | 39.5–42.5% | 1308 |
| Power-SMC (α = 4) on zero-shot CoT, N = 10 / 16, paper's weight-sampled output | 37.0% / 40.5% | 980 / 1426 |
| Power-SMC N = 16, majority vote over particles (particles collapse to ~4 distinct outputs) | 47.5% | 1426 |
| Zero-shot CoT ("Let's think step by step.") | 41.0% | 123 |
| CoT-decoding, k = 10, aggregated confidence | 40.0% | 829 |
| CoT-decoding, k = 10, max confidence | 29.5% | 829 |
| Forced "Let's" / "First," | 33.0% / 31.0% | 113 / 136 |
| Budget forcing (zero-shot CoT + one "Wait") | 30.0% | 340 |
| Repetition penalty 1.2 only | 9.5% | 41 |
| CFG on zero-shot CoT, γ = 1.25 / 1.5 / 2.0 (γ = 1.0 control: 41.0%) | 36.5% / 27.5% / 14.5% | 101–132 |
| Contrastive decoding (dropout-0.2 amateur) on zero-shot CoT, β = 0.25 / 0.5 (β = 0 control: 40.5%) | 31.0% / 25.0% | 126 / 132 |
| Self-consistency without the CoT prefix (n = 10) | 10.0% | 588 |
| DoLa (high / low layers, ± repetition penalty) | 5.0–6.5% | 105–141 |

With n = 200 the standard error is about ±3.5 points. Zero-shot CoT is the cheap baseline the
remaining methods must beat; they run on top of its prefix. **Self-consistency beats it by +10 points**
(paired McNemar p = 0.004; maj@k: k=4 40.5%, k=6 48%, k=8 50.5%, k=10 51%; oracle@10 71%) at ~10x the
tokens: the first "costly method that works", and the target for a rule-based (Hewitt-style) replication. Metrics: `results/screen/olmo2-1b-base/dev/`.

Context for small models: majority vote helps only on questions where the model is right more
than half the time and can hurt otherwise ([When SC Hurts](https://arxiv.org/abs/2611.11201),
[SLM-MUX](https://arxiv.org/abs/2510.05077)); our dev gain is an average over both regimes.

### Test subset (500 GSM8K test problems, `data/gsm8k/test500.jsonl`)

| Method | Acc. |
|---|---|
| Base greedy zero-shot | 9.6% |
| Base 8-shot (from full-test run) | 32.2% |
| Base rule adapter, greedy (no prompt) / + voting (n = 10) | 29.2% / 37.8% |
| Base zero-shot CoT | 37.0% |
| **Base self-consistency (n = 10) on zero-shot CoT, majority** | **43.2%** (+6.2 vs zero-shot CoT, McNemar p = 0.002) |
| … + self-certainty Borda vote / entropy-weighted vote / DeepConf | 43.6–44.4% (≈ majority; the +3 on dev did not hold) |
| … + Think@n (DTR) / EBA | 35.0% / 39.4% |
| oracle@10 | 66.0% |
| SFT / DPO / Instruct greedy (from full-test runs) | 60.4% / 59.0% / 67.8% |

### SFT vs RL: inference compute vs. sparse fork decisions (500 test problems unless noted)

| Samples (T = 0.7, majority vote) | 1 | 4 | 16 | 64 | pass@64 |
|---|---|---|---|---|---|
| OLMo 2 1B SFT | 49.8% | 59.6% | **73.6%** | **75.8%** | 93.8% |
| OLMo 2 1B Instruct (SFT+DPO+RLVR) | 66.6% | 72.4% | 75.2% | 76.6% | 95.6% |
| OLMo 2 7B SFT | 80.4% | **86.6%** | 89.4% | 90.2% | 98.6% |
| OLMo 2 7B Instruct (SFT+DPO+RLVR) | 84.2% | 87.4% | 90.0% | 90.8% | 98.6% |

Greedy: 1B SFT 60.4% / Instruct 67.8%; 7B SFT 74.6% / Instruct 83.2%. At 7B, SFT with only 4 samples (86.6%) passes RL greedy, and RL's edge at 64 samples is 0.6 points. At 1B, SFT with 16-sample voting passes RL greedy; RL's edge shrinks from
17 points (1 sample) to 0.8 (64 samples) — the ability is in SFT, RL mostly makes the first answer good.

Where RL changes SFT (`analysis/rl_delta.py`, SFT's greedy solutions to 300 training problems): the greedy
token differs at only 7.8% of positions (4.4% for the RLVR stage alone), 83–94% of them fork tokens, and
RL's choice is SFT's 2nd choice 72–80% of the time. On dev (200): SFT 72.0%, Instruct 79.0%; taking
Instruct's token **only at forks** (~21% of positions) gives 77.5% — 79% of the gap. Lexical rules derived
from these differences (swap pairs, token-bias tables) all hurt (62.5–69.5%): rules must depend on what
the model computes at a fork, not on word identity (in progress: `analysis/fork_features.py`).

### Positioning and the efficiency angle (2026-11-22, late)

Closest prior work (verified): RL as sparse reranking at high-entropy decision points, with oracle
interventions, down to 1.5B ([Rethinking RL: Sparse Policy Selection](https://arxiv.org/abs/2605.06241));
base models reach RL behaviour with enough sampling budget ([Budgeted Search](https://arxiv.org/abs/2609.01274)).
So "RL fixes a few forks" and "the earlier stage contains RL's answers" are not new by themselves. What is
distinctive here: the full matched Base->SFT->DPO->RL ladder at 1B and 7B, and training-free *compute
efficiency* (reaching RL-level accuracy with fewer samples).

Adaptive stopping on the saved 1B SFT samples (test500, 20 random orders; stop when the leading answer
is ahead by m votes): m = 2 -> 72.4% at 8.0 samples on average (fixed 8: 69.5%); m = 3 -> 74.6% at 13.0
(fixed 32: 74.4%). RL greedy is 67.8%. Next: combine adaptive stopping with fork branching, and at 7B.

Adaptive stopping at both scales (saved n = 64 samples, test500, stop when the leading answer is 2 votes ahead):
7B SFT 88.7% at 3.8 samples on average (7B Instruct greedy 83.2%; 7B Instruct adaptive 89.3% at 3.1);
1B SFT 72.4% at 8.0 samples (1B Instruct greedy 67.8%).

Stage ladder in ReasonMaxxer's format (1B, SFT's greedy solutions to 300 training problems; % of positions where
the later stage's greedy token differs, inside / outside the earlier stage's top-5): SFT→DPO 5.96% / 0.12%;
DPO→Instruct (RLVR alone) 4.31% / 0.05%; SFT→Instruct 7.55% / 0.28%; 83–94% at fork tokens.
(ReasonMaxxer, Qwen2.5-1.5B base→GRPO: 2.09% / 0.02%.) DPO changes more decisions than RLVR; neither stage
chooses tokens outside the earlier model's top-5.

### Fork-contrastive LoRA: results so far (1B, GSM8K)

On SFT (dev): unbounded loss diverged (round 1); tiny learning rates left the model unchanged (round 2); bounded
token-level DPO at lr 1e-4 (round 3): majority 68.0%, gold 70.0%, shuffled 63.0% vs SFT 69.5% — labels carry signal
(shuffled hurts) but no gain over SFT, merged or fork-only (fork-only: gold 72.5%, majority 69.5%, shuffled 66.0% vs
control 72.0%). Full-data (7,273 questions) retraining in progress.

On Base + zero-shot CoT (1,000 training questions, bounded DPO, lr 1e-4, 2 epochs):

| Variant | test500 | full test (1,319) |
|---|---|---|
| Base + CoT baseline | 37.0% | 38.1% |
| shuffled labels (control) | 35.4% | |
| majority labels (label-free) | 37.6% | |
| gold labels, seed 0 / seed 1 | 40.2% / 39.2% | 40.3% (p = 0.11 vs baseline) |
| **RFT baseline: LoRA fine-tuned on the model's own correct samples (gold)** | **45.6%** (dev 51.5%; p < 0.0001 vs baseline) | **43.7%** (p < 0.00001) |
| **Label-free RFT: fine-tuned on samples agreeing with the model's own majority answer** | **41.4%** (dev 47.0%; p = 0.026) | **41.6%** (p = 0.003) |

Gold vs shuffled on test500: +4.8, McNemar p = 0.037. A consistent but small gain (~+2) with gold labels that does not
reach significance against the untrained baseline; the label-free version gives nothing on Base (majority labels were
correct on only 46% of training questions). **The RFT baseline beats the fork-contrastive LoRA by 5.4 points on test500 (same 1,000 questions, same LoRA settings): the fork-local training is not what helps; whole correct self-generated solutions are a much stronger signal.** Label-free (majority) RFT also helps significantly (+4.4), recovering ~70% of what 10-sample self-consistency gives on this setup (43.2%) at 1x inference cost — even though the base model's majority answer was right on only 46% of the training questions.

### Inference-time methods on the full GSM8K test (seahorse, 2026-11-22/23)

`scripts/fulltest_inference.sh`, 1,319 problems, `methods.run`, bf16; each method at its best Base-dev setting (SFT reuses
it, not re-tuned; Base 256 / SFT 512 new tokens). Metrics in `results/fulltest/`; paired McNemar: `analysis/fulltest_paired.py`
(no multiple-comparison correction). RL reference: **Instruct greedy 67.1%**. "Gap closed" = (method - greedy) / (Instruct - greedy).

| 1B SFT (greedy 58.9%) | Acc. | vs greedy | Gap closed | vs Instruct greedy | Tokens |
|---|---|---|---|---|---|
| Self-consistency n=10 + self-certainty Borda 0.5 / 1.0 | 72.2 / 72.0% | +13.3 / +13.1 | 162 / 160% | not tested | 1561 + rescoring |
| CoT-decoding k=10, aggregated | 71.6% | +12.7 | 155% | +4.5 (p<0.001) | 1447 |
| Self-consistency n=10, majority | 70.0% | +11.1 | 135% | +2.9 (p=0.017) | 1561 |
| EAGer theta 1.0, M=10 | 68.1% | +9.2 | 112% | +1.0 (p=0.46) | 1304 |
| Power-SMC N=16, majority / sampled | 60.7 / 60.2% | +1.7 / +1.3 (n.s.) | 21 / 16% | -6.4 / -6.9 | 2122 |
| CFG 1.25 | 60.3% | +1.4 (n.s.) | 17% | -6.8 | 127 (x2 passes) |
| CoT-decoding k=10, max confidence | 59.2% | +0.3 (n.s.) | 4% | -7.9 | 1447 |
| TLSC K=4 | 58.9% | 0.0 | 0% | -8.2 | 542 |
| Contrastive 0.25 / DoLa / budget forcing | 56.3 / 53.2 / 51.8% | -2.7 / -5.7 / -7.1 | <0 | -10.8 / -13.9 / -15.3 | 148 / 151 / 298 |

| 1B Base (CoT greedy 37.3%) | Acc. | vs CoT greedy | Gap closed |
|---|---|---|---|
| Self-consistency n=10 + self-certainty Borda 0.5 / 1.0 | 45.7 / 44.7% | +8.4 / +7.4 | 28 / 25% |
| Self-consistency n=10, majority | 43.1% | +5.8 (p<0.001) | 19% |
| EAGer theta 1.0, M=10 | 42.9% | +5.6 (p<0.001) | 19% |
| Power-SMC N=16, majority / sampled | 41.6 / 35.9% | +4.3 (p=0.002) / -1.4 | 15% / <0 |
| Rule adapter A, vote / greedy | 37.0 / 30.7% | -0.3 / -6.6 | <0 |
| CoT-decoding k=10, aggregated / max | 34.6 / 29.0% | -2.7 / -8.3 | <0 |
| TLSC / CFG 1.25 / contrastive 0.25 | 34.1 / 32.9 / 32.3% | -3.2 / -4.4 / -5.0 | <0 |
| Forced "First," / budget forcing / forced "Let's" | 29.4 / 28.2 / 27.9% | -7.9 / -9.1 / -9.4 | <0 |
| Zero-shot greedy / DoLa (no CoT prefix) | 8.1 / 5.8% | -29.2 / -31.5 | <0 |

Multi-path voting takes SFT to or past RL greedy at ~8-9x its 171 tokens (RL also improves with voting); no single-output
method helps significantly. Base is not closed this way.

### Training-free "global forks" (SSFT idea, 1B SFT; seahorse)

Fixed opening set (`analysis/global_forks*.py`, 500 train questions, 6 paths, 20 CV splits): chosen set 80.6% vs plain
sampling 80.9% vs best single opening x6 81.3%: no gain. First-token branching (top-k first tokens of each question, then
greedy) beat T=0.7 voting on the full test at k=4/6 (66.9 vs 60.7, 68.7 vs 66.1), not at k=8/10 (post hoc). Control
(`analysis/first_token_control.py`, 1,000 other train questions, majority vote, one vLLM pipeline, sampling = mean of 3 seeds):

| Condition | k=1 | k=4 | k=6 | k=8 | k=10 |
|---|---|---|---|---|---|
| First-token forks, greedy | 70.0 | **77.6** | **79.8** | **81.4** | **82.1** |
| First-token forks, T=0.3 / 0.5 / 0.7 | 68.0 / 66.1 / 61.6 | 77.6 / 77.7 / 73.2 | 79.9 / 79.7 / 77.7 | 81.4 / 81.0 / 79.3 | 82.0 / 81.4 / 80.6 |
| Sampling T=0.3 | 68.2 | 75.4 | 78.3 | 79.0 | 79.7 |
| Sampling T=0.5 | 64.1 | 74.7 | 78.2 | 79.9 | 80.9 |
| Sampling T=0.7 | 59.3 | 72.9 | 76.6 | 79.4 | 80.8 |
| Greedy + samples T=0.3 / 0.5 / 0.7 | 68.7 | 75.2 / 75.4 / 75.1 | 75.8 / 77.5 / 78.0 | 77.5 / 79.4 / 79.8 | 79.5 / 80.4 / 80.6 |

Forks (greedy) minus the best sampling temperature [95% bootstrap CI]: k=4 +2.2 [+0.2, +4.2] (T 0.3); k=6 +1.5 [-0.4, +3.6]
(T 0.3); k=8 +1.5 [-0.4, +3.4] (T 0.5); k=10 +1.2 [-0.6, +3.1] (T 0.5). So most of the test-set gap at k=4 (+6.2) was the T=0.7 baseline; what is
left is ~1-2 points, above zero only at k=4. Coverage at 6 paths is about equal (90.7% vs 89-91%). Cost is equal (879 vs
~873 tokens at k=6). Train accuracy is higher than test (SFT may have seen these questions). Not confirmed on test.

### Self-SSFT: global forking tokens trained on the model's own traces (1B SFT, test500)

SSFT (Jia et al., ICLR 2026, [arXiv 2510.05132](https://arxiv.org/abs/2510.05132)) adds tokens `<think1..N>` and
trains them by **set matching**: each question's distinct traces are matched one-to-one to the tags by minimum loss,
and each tag is trained only on its matched trace, so that each tag starts a different reasoning mode. The paper
used 32B models and teacher traces. We test it at 1B with the model's **own** RFT traces (`train/ssft_lora.py`,
`scripts/gcp_ssft.sh`): 8 SFT samples (T 0.7) on 1,500 training questions, up to 4 most-distinct correct
(or majority-agreeing) traces per question, N = 4 tags, LoRA r32 + tag embeddings, 2 epochs. Controls on the same
traces: no tags (= RFT, `rft_*_m4`), random tag assignment (`ssft_gold_random`), and standard RFT on all correct
samples. One vLLM pipeline for all models; equal compute for voting = 4 paths (one greedy path per tag vs 4 samples
at T 0.7); `analysis/ssft_cons.py`.

| Model | Pass@1 (greedy; tagged models: `<think1>`) | Vote@4 | Pass@4 | Distinct answers / 4 |
|---|---|---|---|---|
| SFT | 58.8 | 61.6 | 75.0 | 2.47 |
| RFT, same capped traces (gold) | 58.2 | 61.8 | 76.4 | 2.30 |
| RFT, all correct samples (gold) | 57.2 | 64.0 | 75.0 | 2.23 |
| **Self-SSFT (gold)** | **60.2** | 63.2 | 76.8 | 1.89 |
| Self-SSFT, random tag assignment | 58.0 | 58.8 | 65.2 | 1.29 |
| Self-SSFT (majority, label-free) | 59.4 | 62.2 | 75.0 | 1.89 |
| RFT, same capped traces (majority) | 55.6 | 62.8 | 77.6 | 2.29 |
| Instruct (RL) | 66.6 | 74.4 | 84.8 | 1.84 |

McNemar on Vote@4: Self-SSFT vs RFT (same traces) p = 0.53, vs RFT (all) p = 0.76, **vs random tags p = 0.012**;
label-free Self-SSFT vs label-free RFT p = 0.83. Training: tag usage stayed balanced (~145 traces per tag per
logging window), and 52–59% of questions kept the same tag assignment across epochs.

**Takeaways.** (1) The matching is what makes the tags differ: random assignment leaves the four tags nearly
identical (1.29 distinct answers) and votes significantly worse. (2) No mode collapse. (3) But at 1B on GSM8K the
tags do **not** beat temperature sampling: one greedy path per tag is slightly more accurate per path but less
diverse (1.89 vs 2.30 distinct answers), so Vote@4 is the same as RFT's maj@4. (4) `<think1>` alone is +2.0 over
RFT on the same traces (60.2 vs 58.2), not yet significance-tested. The paper's central claim (learned global forks
diversify better than temperature) does not reproduce at this scale. In this pipeline RFT also did not improve over
SFT greedy, unlike the earlier HF-decoder run (59.8 → 63.4), so the two setups are not directly comparable.

### Overnight results, 2026-11-23 (main machine)

P-values are paired McNemar tests, uncorrected. The GSM8K test set has been used by many configurations, so treat them as
exploratory. Compare numbers only within a block: decoders, precision and batch sizes differ between blocks.

**Contrastive weights from 1B SFT** (`scripts/queue_contrast.sh`; fp32, full test). theta = SFT + RFT + lambda (correct − wrong):
LoRAs trained separately on SFT's own correct and wrong solutions (2,452 each, matched per question). One seed.

| Model | Full test | vs RFT |
|---|---|---|
| SFT | 58.2% | p = 0.005 |
| RFT (7,412 correct solutions) | 61.4% | – |
| RFT + random halves of the same solutions, lambda 0.25 (control) | 61.2% | p = 0.82 |
| correct − wrong only, lambda 2 | 62.4% | p = 0.46 |
| **RFT + (correct − wrong), lambda 1** | **64.1%** | **p = 0.013** (vs control p = 0.007) |
| Instruct (RL) | 68.1% | p < 0.0001 |

Caveats: lambda 1 is the edge of the grid; held-out separates it only weakly from RFT (78.8 vs 77.0, p = 0.32); the control was
tested at its own best lambda (matched lambda 1 on held-out only: 72.8); prior-work search not done (task arithmetic
2212.04089, ExPO 2404.16792 and cDPO 2411.19943 are the closest known).

**RFT on 1B SFT, full test** (`scripts/queue_sft_rft_fulltest.sh`): SFT 59.4; gold 62.9 (p = 0.002); label-free 62.0 (p = 0.028);
Instruct 67.6. Supersedes the test500 reading that label-free RFT gives no gain on SFT.

**Base + CoT, ReasonMaxxer variants, full test** (prompt-only 38.1, RFT on 686 problems 43.7): tamed entropy gate (beta_neg 0.3, lr 5e-5)
44.4; same with 200 problems 40.5 (149-step cap: saw about 1,200 of 3,200 rollouts); RFT on the same 50 problems 42.1; tamed soft fork weights 42.2; original entropy gate 41.5; tau 1.4 nats
40.0; soft fork weights 38.4 / 38.7; hard fork gate 36.0 (checkpoint not verifiable, rerun needed); entropy gate positive-only 35.9
(only its step-50 evaluation is valid). Fork-branch RFT 41.5, with samples 42.7.

**Seeds + 2x2** (`scripts/queue_seeds_2x2.sh`; Base, 3 LoRA-init seeds, fp32, fresh1000 / SVAMP): prompt-only 44.2 / 42.3; RFT-50
46.4 / 52.9; tamed 52.7 / 51.2 (seed-0 scores reproduce exactly from the saved adapters; but RFT-50 is level or ahead on heldout500, 50.2 vs 48.6, and on SVAMP, and the sets disagree for reasons we have not found); EAPO rule 48.2 / 52.3; symmetric 49.2 / 52.4;
uniform positives with low / high negatives 48.6 / 52.0 and 49.1 / 51.3. An offline adaptation of EAPO (2609.33781), not a replication.

**Extrapolation past Instruct** (alpha 1.5, i.e. 0.5 in ExPO's convention): full test 69.9 vs 67.6; on the 819 questions outside test500
70.7 vs 67.2 (p = 0.008); on test500 68.6 vs 68.2.

**Analysis caveat**: a small top-2 margin makes an argmax flip easier almost by definition, so "the margin gate predicts RL's changes
better than entropy" (`results/gate_vs_rl`) is partly circular. bf16 merges also round by about as much as the LoRA change.

### Is the ability already in the base model? (reasoning analogue of Hewitt et al. Table 2)

`analysis/ranking_test.py --natural`: on the 138 dev problems where the base model's own 10
zero-shot-CoT samples include both correct and incorrect solutions, how often does each model
assign higher per-token probability to a correct sample than to an incorrect one (all pairs)?

| Model | Prefers correct reasoning |
|---|---|
| OLMo 2 1B Base | 67.1% |
| SFT | 68.9% |
| DPO | 70.9% |
| Instruct (RLVR) | 72.1% |

The base model already discriminates its correct from its incorrect reasoning nearly as well
as the post-trained models (cf. Hewitt et al.: base 80.4% vs. instruction-tuned 77.4%). Against
another question's solution (Hewitt's exact test) all four models are at 88–94%.

### Future directions

- **Next method candidates (2026-11-23), building on what works (RFT on own correct samples):**
  (a) *Faithful ReasonMaxxer on OLMo* ([arXiv 2605.06241](https://arxiv.org/abs/2605.06241)): advantage-weighted
  loss at ALL high-entropy decision points of full rollouts (positives and negatives), KL anchor elsewhere. Their
  ablation: positive-only (= RFT) gives about half the gain, adding negatives the other half. Our RFT results are
  the positive-only half; our fork-LoRA was a different single-token variant, so the faithful recipe is untested
  on OLMo. Variants: gold vs. majority (label-free) advantages; Base vs. SFT; 1B vs. 7B.
  (b) *Fork-branch self-training*: on training questions, decode greedily, branch at the closest-call forks, and
  fine-tune on the corrected branch (same prefix, one different decision, own continuation) wherever it reaches the
  gold / majority answer and greedy did not; optionally use the failing greedy path as the negative at that fork.
  Minimal, on-policy corrections at the decisions RL changes. Compare to RFT at equal data.
- **Planned method: fork-contrastive self-distillation (LoRA).** Build on ReasonMaxxer
  ([arXiv 2605.06241](https://arxiv.org/abs/2605.06241): LoRA trained only at high-entropy decision points ≈ RL)
  with two changes from our findings: (1) training pairs are *minimal counterfactual pairs* from fork branching
  (same prefix, one different token at a fork), each fork token scored by several sampled continuations;
  (2) labels come from the model's own majority vote (held-out samples), not gold answers. Goal: compress
  self-consistency into one greedy pass (1x) on SFT, compared stage-by-stage with DPO/Instruct. Controls and
  ablations: gold-label version, ReasonMaxxer-style independent rollouts, shuffled labels, random-position forks,
  full-sequence majority distillation; track pass@k / maj@k / diversity for collapse. Closest prior per a
  literature check: ReasonMaxxer (gold labels, independent rollouts), TTRL and self-training with majority
  rewards (sequence-level RL; collapse risk), self-consistency preference optimization; no prior work found
  combining all three ingredients.
- Ideas that failed or were inconclusive at 1B but may work at larger scale are tracked in [SCALE_NOTES.md](SCALE_NOTES.md).

- **Run the winning methods on OLMo 2 1B SFT (and DPO / Instruct).** Majority vote only helps on
  questions the model gets right more than half the time ([SLM-MUX](https://arxiv.org/abs/2510.05077));
  SFT is right far more often than Base (60% vs 9% greedy), so self-consistency and confidence-based
  aggregation may gain more there. Needs chat-template support in `methods/run.py` (plain format only today).
- StepBack ([NeurIPS 2026](https://neurips.cc/virtual/2026/poster/149117); error-localized resampling,
  then majority vote): reported to match self-consistency at fewer tokens on 3B Instruct models —
  an efficiency method, not an accuracy gain; low priority.
- A literature search found no paper showing a method that beats self-consistency at matched budget on
  a <= 2B base model; a positive result there would be new.

Worth checking directly at 1B: whether contrastive decoding and DoLa hurt small models, and
whether DTR (which does not predict correctness on our greedy outputs) works within a problem.

## Running on Colab

`notebooks/colab_baselines.ipynb` runs a 50-question smoke test per model, then the full test
set. Finished runs are skipped, so re-running after a disconnect resumes.

- **VS Code + Google Colab extension:** open the notebook, *Select Kernel > Colab > New Colab
  Server*, pick a GPU. The kernel runs on the Colab machine, so the notebook clones the repo
  there. Put tokens in `.env` (copy `.env.example`; gitignored) and right-click it > *Upload
  to Colab*; without it the notebook prompts for the GitHub token.
  For results on Drive, run *Colab: Mount Google Drive to Server...* first; otherwise they land
  in `/content/results` and are zipped at the end for download from the *Contents* view.
- **Colab web UI:** pick a GPU runtime, add a `GITHUB_TOKEN` secret, run all cells. Results
  are saved to `MyDrive/implicit-adaptation/`.

## Quick start (any GPU machine)

```bash
PY=~/miniconda3/envs/ICCV/bin/python    # temporary env: torch 2.7, transformers 4.57
$PY -m pip install -r requirements.txt

$PY -m tasks.gsm8k --out data/gsm8k     # train (7,273) / dev (200) / test (1,319)
$PY -m evaluation.generate --model olmo2-1b-base --few_shot 8 \
    --out results/olmo2-1b-base/gsm8k/8shot
$PY -m evaluation.generate --model olmo2-1b-sft --out results/olmo2-1b-sft/gsm8k/0shot
$PY -m analysis.summarize results       # one table over all runs
```

Add `--limit 50` to `evaluation.generate` for a quick check. `--dtype auto` uses bf16 on
L4/A100 and float32 on older GPUs (T4, V100).

## Sampling with vLLM (`--backend vllm`)

`evaluation.generate --backend hf` (default) uses HF `model.generate`, which drops to batch size 1 at
`--n_samples 64`. `--backend vllm` runs the same prompts (incl. chat template and few-shot), the same
stops (plain: `STOP_STRINGS`; chat: EOS only), truncation, scoring and output files, with
`SamplingParams(n=n_samples, temperature, top_p=1.0, seed, max_tokens=max_new_tokens)`. Prompts are
tokenized with the HF tokenizer, and the returned token ids are decoded and counted like the HF path, so
`gen_tokens` and `truncated` mean the same thing. vLLM is imported only on that path, so the HF path still
runs in an env without it.

vLLM needs its own env (pins its own torch). On the lab server (driver 535, CUDA 12.2) we use
vLLM 0.9.2 = torch 2.7.0+cu126, which runs through CUDA minor-version compatibility:

```bash
~/miniconda3/bin/conda create -y -n implicit-vllm python=3.11
VPY=~/miniconda3/envs/implicit-vllm/bin/python
$VPY -m pip install --no-cache-dir vllm==0.9.2 "transformers==4.53.3" "huggingface-hub<1.0" \
    "accelerate<1.8" "datasets<3.7" pandas tabulate   # ~8.4 GB
```

(vLLM 0.9.2 breaks with transformers >= 4.54, and in vLLM <= 0.8.x `gpu_memory_utilization` counts
other processes' memory too, so it fails on a shared GPU.)

```bash
# 500-problem test subset, 64 samples per problem; GPU 1 only, vLLM may take 35% of its memory
CUDA_VISIBLE_DEVICES=1 $VPY -m evaluation.generate --backend vllm --model olmo2-1b-sft \
    --subset 500 --n_samples 64 --temperature 0.8 --gpu_memory_utilization 0.35 \
    --out results/olmo2-1b-sft/gsm8k/0shot_sub500_n64_t0.8
GPU=1 bash scripts/gsm8k_sampling.sh    # all 5 configs (base 0/8-shot, SFT, DPO, Instruct)
```

- `--subset N` picks a fixed random subset of the split: `random.Random(0).sample(range(len(split)), N)`,
  sorted, so the same 500 test problems for every model (`--subset_seed` changes the seed; keep 0).
  It is applied before `--limit`. Each generations row stores its `idx` in the full split.
- `--gpu_memory_utilization` (default 0.35) is the share of the GPU's total memory this vLLM instance may
  use (weights + KV cache). Check `nvidia-smi` first; the 1B models need ~3 GB of it for weights.
- vLLM flags: `--max_model_len` (default 4096), `--enforce_eager` (no CUDA graphs). `--batch_size` is HF only.
- "Error in memory profiling" at startup means another process freed GPU memory while vLLM was
  profiling; `generate.py` retries startup up to 3 times.
- Greedy vLLM matches greedy HF up to bf16 noise: on the first 100 test problems it agrees with HF as often
  as HF at batch 64 agrees with HF at batch 1 (see ROADMAP).

## Conventions

- Base models: `Question: {q}\nAnswer:`, generation cut at a blank line. Post-trained models:
  their chat template, with the question plus "Solve the problem step by step. End your
  response with "The answer is N."", stopping at EOS only. Reference solutions are the GSM8K
  rationale on one line, without `<<calc>>` notes, ending in `Therefore, the answer is N.`
- Two accuracies: **strict** (`the answer is N`) and **lenient** (first `answer ...`,
  `#### N` or `\boxed{N}`, else the last number). The gap shows how much a model loses to format alone.
- Numbers are normalized before comparing (`$`, commas, trailing `.0`).
- Raw generations are saved, so a parser fix means re-scoring, not re-generating.
