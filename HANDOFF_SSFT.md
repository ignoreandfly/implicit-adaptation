# Handoff: SSFT global forking tokens, inference-time test (8-GPU machine)

For a Claude Code session on the 8×GPU server (seahorse, 8× A6000 48 GB). Read this file, `README.md` and
`RUNBOOK.md` §1 (setup). Needed today, so follow the priority order.

## Why
Our mentor asked us to try **SSFT** at inference time before considering it for our method. Paper:
"Training LLMs to Reason in Parallel with Global Forking Tokens", Jia et al., ICLR 2026,
[arXiv 2510.05132](https://arxiv.org/abs/2510.05132), code at [github.com/Sheng-J/SSFT](https://github.com/Sheng-J/SSFT).
- It adds N = 6 special tokens `<think1>`…`<think6>` and trains them (set-based SFT with Hungarian matching) so that each one triggers a
  different reasoning mode.
- At inference: prompt once per token and majority-vote (**Cons@6**), or use `<think1>` alone (**Pass@1**).
- Their claim: this gives better diversity at equal accuracy than temperature sampling.
- Their evaluations are AIME / MATH-500 / GPQA / LiveCodeBench at 32B. **There is no GSM8K result and no small model.**
- There is no training-free version, so "inference time" means: **use their released trained checkpoint, no training by us**.

Two questions:
- **A. Does SSFT's mechanism help on GSM8K?** Released 32B checkpoints, inference only.
- **B. Can natural forks give the same benefit without training?** Our OLMo 2 1B fork-branching vs temperature vs
  first-token branching, at **equal compute**.

## Setup
```bash
git pull                          # analysis/ssft_infer.py and analysis/equal_compute.py are new
# envs and data as in RUNBOOK.md §1 (the implicit-vllm env for part A; the ICCV-style hf env for part B);
# data/gsm8k/test500.jsonl must exist
nvidia-smi                        # use only free GPUs; never kill other users' jobs
export VPY=<vllm env python>  HPY=<hf env python>
# A downloads ~2 x 65 GB into the HF cache; set HF_HUB_CACHE to a disk with >= 150 GB free
```
Check first: the baseline model's tokenizer and model card. Does `sft_mixed_32b_thinktags_GAS4` contain the `<thinkN>`
tokens, and what prompt suffix does the authors' eval use for it? See `lighteval/src/lighteval/models/vllm/vllm_model.py`
in their repo. If it uses `<|im_start|><think1>\n`, run the baseline with `--mode repeat --tag 1`. Otherwise use `--mode plain`.

## A. Released SSFT-32B vs its matched SFT baseline (GSM8K test500)
Prompt: chat template + "Please reason step by step, and put your final answer within \boxed{}." + forking-token suffix
`<|im_start|><think{i}>\n`, exactly as in the authors' LightEval code. Sampling settings are the paper's: T 0.7, top-p 0.95.
Each run below uses 2 GPUs (`--tp 2`). Run A1–A3 in parallel on 6 GPUs, then A4.

| id | command (prefix: `CUDA_VISIBLE_DEVICES=<2 gpus> $VPY -m analysis.ssft_infer --split test500 --tp 2`) | what it measures |
|---|---|---|
| A1 | `--model shengjia-toronto/ssft-32B-N6 --mode tags --n_tags 6 --temperature 0.7 --out results/ssft/ssft32b_tags6_T0.7` | the paper's Cons@6 (one path per forking token) |
| A2 | `--model shengjia-toronto/ssft-32B-N6 --mode repeat --tag 1 --n 6 --temperature 0.7 --out results/ssft/ssft32b_think1x6_T0.7` | **key control**: same model, same compute, temperature-only diversity |
| A3 | `--model shengjia-toronto/sft_mixed_32b_thinktags_GAS4 --mode <plain or repeat --tag 1> --n 6 --temperature 0.7 --out results/ssft/sftmixed32b_x6_T0.7` | the paper's baseline, maj@6 |
| A4 | `--model shengjia-toronto/ssft-32B-N6 --mode tags --n_tags 6 --temperature 0 --out results/ssft/ssft32b_tags6_greedy` | greedy per token: diversity from the forking tokens alone, no temperature |

Then score all four at equal compute:
`$HPY -m analysis.equal_compute results/ssft/*/test500.generations.jsonl --ks 1 2 3 4 5 6 --out results/ssft/equal_compute.json`
Each metrics file also has `acc_per_tag` (Pass@1 per forking token), the truncation rate and tokens per path.

How to read the results:
- A1 > A2 at k = 6: the forking tokens give better diversity than temperature (the paper's mechanism).
- A1 > A3: SSFT beats standard SFT on GSM8K.
- A4 ≈ A1: the tokens alone diversify.
- GSM8K may be too easy at 32B (ceiling ~95%). If everything sits above 93%, say so; that is a result too.

Budget: these are long-reasoning models. `--max_tokens 8192`, and check the truncation rate. Expect ~1 h per run with
2 GPUs; test with `--limit 20` first (5 min).

## B. Training-free fork branching vs temperature at equal compute (OLMo 2 1B SFT, test500)
Same HF pipeline for all (`methods.run`, chat format). One GPU each, ~30–60 min each. Run on the 2 GPUs left over from A.
```bash
cd <repo>; export CUDA_VISIBLE_DEVICES=<gpu>
# B1 natural forks: greedy + branches at the 7 smallest top-2 margins (gives k = 1..8 in one run)
$HPY -m methods.run --method fork_mbr --model olmo2-1b-sft --format chat --split test500 --fm_forks 7 --fm_min_pos 0 \
  --batch_problems 4 --out results/equal_compute/sft_fork_F7
# B2 "deep" forks only: skip the first 30 response tokens
$HPY -m methods.run --method fork_mbr --model olmo2-1b-sft --format chat --split test500 --fm_forks 7 --fm_min_pos 30 \
  --batch_problems 4 --out results/equal_compute/sft_fork_F7_deep30
# B3 temperature sampling, 8 paths
$HPY -m methods.run --method sample --model olmo2-1b-sft --format chat --split test500 --n 8 --temperature 0.7 \
  --batch_problems 4 --out results/equal_compute/sft_sample_T0.7_n8
# B4 first-token branching (a "global fork" at position 0; CoT-decoding, Wang & Zhou 2024), 8 paths
$HPY -m methods.run --method cot_decoding --model olmo2-1b-sft --format chat --split test500 --k 8 \
  --batch_problems 4 --out results/equal_compute/sft_firsttoken_k8
$HPY -m analysis.equal_compute results/equal_compute/*/test500.generations.jsonl --ks 1 2 4 6 8 \
  --out results/equal_compute/summary.json
```
Check first: whether `cot_decoding` supports `--format chat` (`args.fmt`). If not, drop B4 or add chat support, as was done
for `fork_mbr`.

Reference (main machine, the B1 setting with 8 forks): vote 56.6% → 58.6% → 59.4% → 61.0% at k = 1/4/6/8 paths, pass@8 77.6%.
Reading the results: B1/B2 > B3 at equal k means natural forks beat temperature (the SSFT premise, training-free). B3 > B1 means
temperature diversity is enough at 1B on GSM8K. Compare A and B qualitatively: learned global forks at 32B vs natural local forks at 1B.

## Reporting
- Paired McNemar tests on the per-question vote outcomes at k = 6: A1 vs A2, A1 vs A3, B1 vs B3.
  Recompute them from the jsonl files as in `README.md`.
- Add a short "SSFT at inference time" section to `README.md` with the two equal-compute tables.
- Commit only `results/ssft/**/*.json`, `results/equal_compute/**/*.json` and README/script changes. Generations jsonl
  files are gitignored.
- Always `git pull --rebase` first; the main machine pushes to the same repo.
```bash
git pull --rebase && git add results/ssft results/equal_compute README.md analysis scripts && git commit -m "..." && git push
```

## Pitfalls
- Launch long runs with `setsid nohup … > log 2>&1 < /dev/null &`. Wait on PIDs or files, not on `pgrep -f`
  patterns that match your own shell.
- vLLM start-up can fail when other users' memory changes. Retry, or lower `--gpu_memory_utilization` (e.g. 0.8).
- `analysis/ssft_infer.py` is new and untested on GPU, so start with `--limit 20`. It verifies on the text after `</think>`,
  and on the last `\boxed{}` when present. Spot-check 5 outputs by hand.
