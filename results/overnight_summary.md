# Overnight summary (GSM8K, 500-problem test subset)

## Greedy

| Model | Greedy (vLLM / generate pipeline) | Greedy (manual decode loop, batch 16) |
|---|---|---|
| 1B SFT | 0.604 (full-test run) | 0.604 |
| 1B Instruct | 0.678 (full-test run) | 0.682 |
| 7B SFT | 0.746 | |
| 7B Instruct | 0.832 | |

## Sampling + majority vote (T = 0.7)

- **olmo2-1b-sft**: maj@1 0.498, maj@4 0.596, maj@16 0.736, maj@64 0.758 | pass@1 0.528, pass@4 0.773, pass@16 0.890, pass@64 0.938
- **olmo2-1b-dpo**: maj@1 0.574, maj@4 0.670, maj@16 0.736, maj@64 0.734 | pass@1 0.583, pass@4 0.792, pass@16 0.891, pass@64 0.948
- **olmo2-1b-instruct**: maj@1 0.666, maj@4 0.724, maj@16 0.752, maj@64 0.766 | pass@1 0.675, pass@4 0.826, pass@16 0.909, pass@64 0.956
- **olmo2-7b-sft**: maj@1 0.804, maj@4 0.866, maj@16 0.894, maj@64 0.902 | pass@1 0.764, pass@4 0.927, pass@16 0.973, pass@64 0.986
- **olmo2-7b-instruct**: maj@1 0.842, maj@4 0.874, maj@16 0.900, maj@64 0.908 | pass@1 0.833, pass@4 0.935, pass@16 0.971, pass@64 0.986

## FOBAR backward verification (label-free selection among pool answers)

| Pool | majority | backward only | FOBAR | rescue / harm (FOBAR vs majority) | oracle |
|---|---|---|---|---|---|
| olmo2-1b-sft_fork_k8 | 0.608 | 0.644 | 0.646 | 30 / 11 | 0.786 |
| olmo2-1b-sft_sampling_k16 | 0.736 | 0.658 | 0.686 | 12 / 37 | 0.908 |
| olmo2-1b-sft_sampling_k8 | 0.678 | 0.652 | 0.676 | 13 / 14 | 0.842 |

## Fork-gated MBR (greedy + 2nd-token branches at the F closest-call forks)

- **olmo2-1b-sft**: greedy 0.566; F2 0.590 (+14/-2, 2.4x), F4 0.600 (+19/-2, 3.6x), F6 0.608 (+22/-1, 4.9x), F8 0.608 (+22/-1, 6.2x); oracle 0.786
- **olmo2-7b-sft**: not available
