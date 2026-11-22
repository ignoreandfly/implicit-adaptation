# Ideas to revisit at larger scale

Methods and ideas that did **not** work (or were inconclusive) on OLMo 2 1B, but have a plausible
reason to behave differently at 7B (OLMo 2 7B also releases Base / SFT / DPO / Instruct) or larger.
A failure at 1B is a data point about scale, not a verdict on the idea. Keep each entry with the
1B evidence and the hypothesis for why scale could change it.

Last updated: 2026-11-22.

| Idea | 1B result (where) | Why it might work at scale | What to test at 7B |
|---|---|---|---|
| Deep-thinking ratio / settling depth (Chen et al. 2026), sequence-level and per-fork | No signal: AUROC 0.40–0.50 across stages; Think@n selection 35% vs 43% majority (test500); per-fork settling depth AUROC 0.50, almost every token settles in the last 1–2 of 16 layers | Logit lens is uninformative in a 16-layer model; the paper's r≈0.8 was at 20B–120B with 36+ layers, where intermediate layers form usable predictions | DTR AUROC on 7B stages; per-fork settling depth (`analysis/fork_features.py`) |
| DoLa (layer contrast) | 5–6.5% vs 9.5% greedy (dev) | Same reason: early/middle layers of a 1B model are not yet semantically formed | DoLa on 7B Base/SFT with the zero-shot CoT prefix |
| Contrastive decoding | Dropout-amateur version hurts monotonically (40.5% → 25% at β = 0.5, dev) | The original method needs a *smaller same-family* amateur; at 7B, **OLMo 2 1B is a natural amateur** (same tokenizer) | 7B expert − 1B amateur, β ∈ {0.25, 0.5} |
| Classifier-free guidance on the prompt | Hurts monotonically (41% → 14.5% at γ = 2, dev) | Paper reports gains for small models too, but on other tasks; larger models may tolerate the sharpening | γ ∈ {1.1, 1.25} on 7B |
| Power sampling (Power-SMC, α = 4) | 37–40.5% vs 41% zero-shot CoT; particles collapse to ~4 distinct outputs (dev) | Sharpening helps when the base distribution's mode is good; at 7B the mode is better and there are more distinct good paths | Power-SMC N = 16 on 7B Base; distinct-output count |
| Confidence-weighted voting (self-certainty Borda, DeepConf, entropy-weighted) | +3 on dev but ≈ majority on test (43.6–44.4% vs 43.2%) | Calibration improves with scale; the signal may separate good samples better | Rescore 7B samples with `methods/rescore.py` |
| Embedding-based agreement (EBA) | Below majority (38.5–42.5% vs 51%, dev) | Hidden-state geometry of 1B is noisy; paper used 8B+ | Rescore 7B samples |
| Budget forcing ("Wait") | Hurts (30% vs 41%, dev): base model loops / talks itself out of correct answers | Requires genuine self-correction ability, which emerges with scale | 7B SFT/Instruct with one "Wait" |
| Thought-level self-certainty, EAGer, CoT-decoding | Tie or below zero-shot CoT / voting at equal budget (dev) | Search over steps pays off when the model's own scores rank steps well (better calibrated at scale) | Same scripts on 7B SFT |
| Rule-based Base adapter (Hewitt-style) | 29% test (35% dev), below the zero-shot CoT prompt; first-token rule = soft prompt | At 7B the base model may follow subtler rules; also test the adapter **on top of** the CoT prompt | Adapter on 7B Base, with and without the CoT prefix |
| Lexical "RL rules" on SFT (swap pairs, token-bias tables from Instruct − SFT) | All hurt (62.5–69.5% vs 72%, dev) | Probably a dead end as lexical rules at any scale; keep only as a baseline | — |
| Single-pass fork rules from general features (post-choice entropy / margin) | Weak signal: AUROC ≈ 0.55 | Signals may sharpen with scale | `analysis/fork_features.py` on 7B SFT samples |
| **Cross-scale proxy-tuning of the RL stage** | not run | Apply the 1B (Instruct − SFT) logit difference to **7B SFT** at decode time, i.e. "RL at 7B without RL" (open per literature) | Requires 7B SFT + 1B SFT + 1B Instruct on one GPU (~21 GB bf16) or 8-bit 7B |
| Fork-only oracle (Instruct's token at SFT forks) | 79% of the SFT→RL gap from ~21% of positions (dev) | Check whether RL's changes are even sparser / more fork-concentrated at 7B | `analysis/rl_delta.py` + oracle on 7B pair |

## Promising at 1B, to carry to 7B regardless
- Self-consistency on zero-shot CoT (Base) and SFT + voting vs RL (`scripts/gsm8k_sampling.sh`).
- The "SFT contains RL-level answers" curves (maj@k / pass@k per stage).
- The natural-negative ranking test (`analysis/ranking_test.py --natural`).
