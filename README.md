# Implicit Adaptation below 3B Parameters

COMS 4705 final project. Hewitt et al. ("Instruction Following without Instruction Tuning")
showed that a small hand-written program that edits the next-token distribution, with no
training, can make a base model follow instructions. We ask whether that kind of
rule-based adapter, combined with inference-time scaling, can bring out step-by-step
reasoning in models of 1B parameters or fewer, and how close it gets to models that went
through SFT and RL.

We evaluate on GSM8K first. Task list and open decisions: [ROADMAP.md](ROADMAP.md).

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
evaluation/  generate.py (HF generate + scoring), metrics.py (accuracy, pass@k, maj@k, format, steps, length)
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
| Think@n (vote among the top half of samples by deep-thinking ratio) | [Chen et al. 2026](https://arxiv.org/abs/2602.13517) | 20B–120B | running (`methods/rescore.py`) |
| Fork-aware temperature sampling (our adaptation of the fork-token finding) | idea from [Wang et al. 2025](https://arxiv.org/abs/2506.01939) | — | running; to be replaced by EAGer |

### Next (from the literature)

| # | Method | Paper | Shown on | Cost |
|---|---|---|---|---|
| 1 | Self-certainty best-of-N (Borda-weighted vote) | [Kang et al., NeurIPS 2025](https://arxiv.org/abs/2502.18581) | reasoning LLMs, with CoT | rescoring only (implemented) |
| 2 | DeepConf offline (filter by local/lowest-window confidence, then vote) | [Fu et al., ICLR 2026](https://arxiv.org/abs/2508.15260) | Qwen3, GPT-OSS | rescoring only (implemented) |
| 3 | Classifier-free guidance on the prompt | [Sanchez et al., ICML 2024](https://arxiv.org/abs/2306.17806) | Pythia, GPT-2, LLaMA (incl. small); GSM8K | low (implemented, running) |
| 4 | Distillation contrastive decoding (dropout amateur, single model); original contrastive decoding reported to hurt small models | [Phan et al. 2024](https://arxiv.org/abs/2402.14874); [O'Brien & Lewis 2023](https://arxiv.org/abs/2309.09117) | LLaMA; GSM8K | low (implemented, running) |
| 5 | EM-INF (per-step logit adjustment that lowers entropy) | [Agarwal et al., NeurIPS 2025](https://arxiv.org/abs/2505.15134) | Qwen 7B/32B | low–medium |
| 6 | EAGer / entropy-gated branching (branch only at high-entropy tokens) | [EAGer 2025](https://arxiv.org/abs/2510.11170); [Entropy-gated branching](https://arxiv.org/abs/2503.21961) | small LLMs, +4.6% over greedy | medium |
| 7 | Thought-level self-certainty maximization (pick the most certain continuation per step) | [arXiv 2603.07159](https://arxiv.org/abs/2603.07159) | GSM8K/MATH500; up to 4× greedy on Qwen-1.5B | medium |
| 8 | Power sampling (MCMC on the sharpened base distribution; base matches GRPO) | [Karan & Du, ICLR 2026](https://arxiv.org/abs/2510.14901); cheaper: [token-level](https://arxiv.org/abs/2601.21590), [entropy-guided](https://arxiv.org/abs/2606.09926) | Qwen2.5 base models; MATH500 | high (needs vLLM) |
| 9 | CoT steering vector (add a "think step by step" direction to the residual stream) | [Zhang & Viteri 2024](https://arxiv.org/abs/2409.14026); SAE-free: [2505.15634](https://arxiv.org/abs/2505.15634) | Llama3-8B, Mistral-7B | low–medium |
| 10 | φ-Decoding (foresight sampling) | [Xu et al., ACL 2025](https://arxiv.org/abs/2503.13288) | LLaMA3.1-8B, +14% over CoT | high |
| — | Hewitt et al. rule-based adapter | [Hewitt et al. 2024](https://arxiv.org/abs/2409.14254) | Llama-2 7B, OLMo 7B | after the above |

### Screening results so far (OLMo 2 1B Base, GSM8K dev, 200 problems, lenient accuracy)

| Method | Acc. | Tokens / problem |
|---|---|---|
| Greedy zero-shot (baseline) | 9.5% | 40 |
| Greedy 8-shot (prompting reference) | 38.5% | 73 |
| **Self-consistency on zero-shot CoT** (n = 10, T = 0.7, majority vote) | **51.0%** | 1308 |
| Zero-shot CoT ("Let's think step by step.") | 41.0% | 123 |
| CoT-decoding, k = 10, aggregated confidence | 40.0% | 829 |
| CoT-decoding, k = 10, max confidence | 29.5% | 829 |
| Forced "Let's" / "First," | 33.0% / 31.0% | 113 / 136 |
| Budget forcing (zero-shot CoT + one "Wait") | 30.0% | 340 |
| Repetition penalty 1.2 only | 9.5% | 41 |
| CFG on zero-shot CoT, γ = 1.25 / 1.5 (γ = 1.0 control: 41.0%) | 36.5% / 27.5% | 101 / 103 |
| Self-consistency without the CoT prefix (n = 10) | 10.0% | 588 |
| DoLa (high / low layers, ± repetition penalty) | 5.0–6.5% | 105–141 |

With n = 200 the standard error is about ±3.5 points. Zero-shot CoT is the cheap baseline the
remaining methods must beat; they run on top of its prefix. **Self-consistency beats it by +10 points**
(paired McNemar p = 0.004; maj@k: k=4 40.5%, k=6 48%, k=8 50.5%, k=10 51%; oracle@10 71%) at ~10x the
tokens: the first "costly method that works", and the target for a rule-based (Hewitt-style) replication. Metrics: `results/screen/olmo2-1b-base/dev/`.

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

## Conventions

- Base models: `Question: {q}\nAnswer:`, generation cut at a blank line. Post-trained models:
  their chat template, with the question plus "Solve the problem step by step. End your
  response with "The answer is N."", stopping at EOS only. Reference solutions are the GSM8K
  rationale on one line, without `<<calc>>` notes, ending in `Therefore, the answer is N.`
- Two accuracies: **strict** (`the answer is N`) and **lenient** (first `answer ...`,
  `#### N` or `\boxed{N}`, else the last number). The gap shows how much a model loses to format alone.
- Numbers are normalized before comparing (`$`, commas, trailing `.0`).
- Raw generations are saved, so a parser fix means re-scoring, not re-generating.
