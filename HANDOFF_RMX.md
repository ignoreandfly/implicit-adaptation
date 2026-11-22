# Handoff: where should self-training put its signal? (OLMo 2 1B Base, GSM8K)

Instructions for a Claude Code session on a second GPU machine. Read this file, then `README.md` (results so far)
and `RUNBOOK.md` section 1 (setup). Work in this repo; push only small result files (see "Reporting").

## Context in five lines
- Goal: a training-light method that moves OLMo 2 1B toward the RL model (Instruct) on GSM8K, with a novelty claim.
- What works: **RFT**, i.e. LoRA fine-tuning on the model's own correct samples. Base + zero-shot CoT: 37.0 → 45.6%
  on test500, and 38.1 → 43.7% on the full test (p < 1e-5).
- ReasonMaxxer (arXiv 2605.06241: advantage-weighted loss at high-entropy "decision" tokens + KL elsewhere) has NOT
  beaten RFT on OLMo so far: 37.6–39.6% test500. A fair-recipe run is in progress on the main machine (`scripts/queue_rmx_fair.sh`). **Do not duplicate it.**
- Our analysis: 83–94% of the token-level changes from post-training fall on **forks**, tokens where the model's top-2
  probabilities are close. Hypothesis: **training only at fork tokens keeps (or beats) RFT's gain while touching ~5% of tokens.**
- This machine tests that hypothesis with RFT as the anchor. Everything is the same except which tokens get the loss.

## The experiment (`scripts/rmx_gate_rft.sh`)
Same data (1,000 training questions × 8 samples, T 0.7), same optimizer (LoRA r32 on q/k/v/o, lr 1e-4, 2 epochs).
Correct solutions only, weight 1 (positive-only, so nothing can drift). The non-trained positions get a KL anchor.

| run | tokens with the loss | purpose |
|---|---|---|
| `rft_ref` | all (train/rft_lora.py) | the anchor, re-trained on this machine's samples |
| `all` | all, through train/decision_lora.py | sanity check: should match `rft_ref` |
| `margin05` | 5% with the smallest top-2 margin (our fork criterion) | **the hypothesis** |
| `entropy05` | 5% with the highest entropy (ReasonMaxxer's criterion) | the competing criterion at equal sparsity |
| `margin20` | 20% smallest margin | sparsity check |

Checkpoints every 150 optimizer steps are scored on `heldout500` (training questions 6500–6999, which no run trains on).
The best one is evaluated **once** on the full test (1,319). The script prints a progress bar (tqdm) for each step.

## How to run
```bash
git pull
# setup: RUNBOOK.md section 1 (the hf/peft env + data/gsm8k/*.jsonl). The vLLM env is NOT needed here.
nvidia-smi                                  # pick a GPU with >= 14 GB free; never kill other users' jobs
export HPY=<path to the hf/peft env python>
# the full run (~6-8 GPU-hours; resumable: rerun the same command after any crash).
# It first samples the training data (~40 min) and the prompt-only references, then trains and evaluates each run.
mkdir -p logs
GPU=<idx> HPY=$HPY setsid nohup bash scripts/rmx_gate_rft.sh > logs/rmx_gate_rft.log 2>&1 < /dev/null &
tail -f logs/rmx_gate_rft.log               # per-step progress in logs/rmx_gate_rft/*.log

# optional smoke test once the samples exist (~3 min): 3 optimizer steps on 15 questions
CUDA_VISIBLE_DEVICES=<idx> $HPY -m train.decision_lora \
  --gens results/screen/olmo2-1b-base/train/sample_prefix-let-s-think-step-by-step_T0.7_n8 \
  --model olmo2-1b-base --prefix " Let's think step by step." --rft_weights --n_problems 0 --trim 1 \
  --max_steps 3 --epochs 1 --accum 2 --shuffle --save_every 2 --gate margin --share 0.05 --limit 15 --out /tmp/smoke_gate
```
Priority, if time is short: `RUNS="margin05 entropy05 rft_ref"`, then `all margin20`.

**Two GPUs on this machine?** Split the runs: start the script once with `GPU=a RUNS="rft_ref margin05 margin20"` and
once with `GPU=b RUNS="all entropy05"`. Start the second only after the first has finished sampling, so both don't sample:
wait for `results/screen/olmo2-1b-base/train/sample_prefix-let-s-think-step-by-step_T0.7_n8/train.generations.jsonl`.
Use two identical GPU models if possible, and keep all of these runs on this one machine (one pipeline).

## Second job (optional, another GPU): does RL-step extrapolation hold at 7B?
New result on the main machine (full test, same pipeline). Training-free weight extrapolation past the RL model,
theta = DPO + alpha·(Instruct − DPO), gives **69.9% at alpha = 1.5 vs 67.6% for 1B Instruct (paired p = 0.020)**.
alpha = 1.5 was chosen on dev before the test was run; dev values were 78.0 / 78.5 / 80.5 / 82.5 / 79.0 / 76.5 for
alpha = 0.5 / 1 / 1.25 / 1.5 / 1.75 / 2. Extrapolating SFT→DPO only hurts, so the effect is specific to the RL step.
Does it hold at 7B?
```bash
# needs ~100 GB CPU RAM (three fp32 7B state dicts) and a GPU with >= 20 GB free; models download to the HF cache (~60 GB)
# 1) choose alpha on dev (200 questions)
CUDA_VISIBLE_DEVICES=<idx> $HPY -m analysis.weight_sweep --gpu <idx> --a olmo2-7b-dpo --b olmo2-7b-instruct \
  --alphas 1 1.25 1.5 2 --split dev --tag 7b_dpo__instruct --scratch <fast disk>/ws7b_tmp
# 2) the dev-best alpha and alpha = 1 (Instruct) on the FULL test, same pipeline
$HPY -m analysis.weight_sweep --gpu <idx> --a olmo2-7b-dpo --b olmo2-7b-instruct --alphas <best> 1 --split test \
  --tag 7b_dpo__instruct_fulltest --scratch <fast disk>/ws7b_tmp
```
`analysis/weight_sweep.py` evaluates with `methods.run` (HF greedy, batch 2). A 7B full test is slow, a few hours.
If it is too slow, run step 2 with `--split test500` first. Report the paired McNemar test, as for 1B.
If RAM is short, tell the main session; a streaming (per-tensor) version of the sweep would be needed.

## Pitfalls (learned the hard way; see RUNBOOK §4)
- Launch with `setsid nohup … &`. Wait on a PID or a file. Never use `pgrep -f`/`pkill -f` patterns that match your own shell.
- On a shared GPU, memory changes between checking and loading. The script waits for free memory and retries evals.
- Compare runs only within this machine's pipeline. 1B greedy decoding differs by a few points across code paths
  and batch sizes. That's why `rft_ref` and the prompt-only reference are re-run here.
- Judge by the **full test**, not heldout500 or dev (SE ≈ 3 points on 200–500 questions).

## Reporting
When runs finish, compute paired McNemar tests on the full test (`best_test/test.generations.jsonl` of each run):
`margin05` vs `rft_ref`, `margin05` vs `entropy05`, and each vs prompt-only. Example in `README.md` (paired tests).
Then add a short table to `README.md` under the fork-LoRA / RFT results. Commit **only** `results/rmx_gate_rft/**/*.json`
and `*.txt` (generations jsonl and checkpoints are ignored or too large), the README, and any script fixes:
```bash
git pull --rebase && git add results/rmx_gate_rft README.md scripts && git commit -m "..." && git push
```
Another machine pushes to the same repo, so always `git pull --rebase` first and merge README edits rather than overwrite them.

## What counts as a result
- `margin05` ≥ `rft_ref` (not significantly worse) while training on ~5% of tokens: sparse fork-only training is enough.
- `margin05` > `rft_ref`, or `margin05` > `entropy05`, with p < 0.05: the fork criterion matters. This is the novelty claim.
  Then re-run the winner with `--seed 1` before anyone writes it up.
- Everything ≈ prompt-only: positive fork-only training doesn't carry the signal. That's still worth reporting.
