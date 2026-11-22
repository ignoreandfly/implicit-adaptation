# Fork-contrastive self-distillation, OLMo 2 1B SFT (GSM8K)

| Variant | dev greedy | test500 greedy | test500 maj@16 | test500 pass@16 | train forks |
|---|---|---|---|---|---|
| sft_baseline | 0.695 | 0.594 | 0.732 | 0.902 | - |
| major_lr1e-4 | 0.530 | 0.438 | 0.490 | 0.686 | 1547 |
| major_lr3e-4 | 0.220 | - | 0.286 | 0.508 | 1547 |
| gold_lr1e-4 | 0.390 | 0.308 | - | - | 1588 |
| shuffled_lr1e-4 | 0.420 | 0.392 | 0.474 | 0.722 | 1547 |

Reference (test500): SFT greedy 0.604, Instruct greedy 0.678/0.682, SFT maj@16 0.736.
