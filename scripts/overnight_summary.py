"""Collect the overnight results into one markdown summary (stdout)."""

import json
import os

R = "results"


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def curve(p):
    m = load(p)
    if not m:
        return "not available"
    ks = [k for k in (1, 4, 16, 64) if f"maj@{k}" in m]
    maj = ", ".join(f"maj@{k} {m[f'maj@{k}']:.3f}" for k in ks)
    pas = ", ".join(f"pass@{k} {m[f'pass@{k}_lenient']:.3f}" for k in ks if f"pass@{k}_lenient" in m)
    return f"{maj} | {pas}"


def acc(p, key="acc_lenient"):
    m = load(p)
    return f"{m[key]:.3f}" if m and key in m else "not available"


print("# Overnight summary (GSM8K, 500-problem test subset)\n")
print("## Greedy\n")
print("| Model | Greedy (vLLM / generate pipeline) | Greedy (manual decode loop, batch 16) |")
print("|---|---|---|")
print(f"| 1B SFT | 0.604 (full-test run) | {acc(f'{R}/screen/olmo2-1b-sft/test500/rl_rules_greedy_bs32/test500.metrics.json')} |")
print(f"| 1B Instruct | 0.678 (full-test run) | {acc(f'{R}/screen/olmo2-1b-instruct/test500/rl_rules_greedy_bs32/test500.metrics.json')} |")
print(f"| 7B SFT | {acc(f'{R}/olmo2-7b-sft/gsm8k/0shot_sub500_greedy/test.metrics.json')} | |")
print(f"| 7B Instruct | {acc(f'{R}/olmo2-7b-instruct/gsm8k/0shot_sub500_greedy/test.metrics.json')} | |")

print("\n## Sampling + majority vote (T = 0.7)\n")
for m in ("olmo2-1b-sft", "olmo2-1b-dpo", "olmo2-1b-instruct", "olmo2-7b-sft", "olmo2-7b-instruct"):
    print(f"- **{m}**: {curve(f'{R}/{m}/gsm8k/0shot_sub500_n64_t0.7/test.metrics.json')}")

print("\n## FOBAR backward verification (label-free selection among pool answers)\n")
print("| Pool | majority | backward only | FOBAR | rescue / harm (FOBAR vs majority) | oracle |")
print("|---|---|---|---|---|---|")
for f in sorted(os.listdir(f"{R}/fobar")) if os.path.isdir(f"{R}/fobar") else []:
    s = (load(f"{R}/fobar/{f}") or {}).get("summary")
    if s:
        print(f"| {f[:-5]} | {s['majority']:.3f} | {s['backward']:.3f} | {s['fobar']:.3f} | "
              f"{s.get('rescue_fobar')} / {s.get('harm_fobar')} | {s['oracle']:.3f} |")

print("\n## Fork-gated MBR (greedy + 2nd-token branches at the F closest-call forks)\n")
for m in ("olmo2-1b-sft", "olmo2-7b-sft"):
    s = load(f"{R}/screen/{m}/test500/fork_mbr_F8_minpos0/test500.metrics.json")
    if not s:
        print(f"- **{m}**: not available")
        continue
    row = ", ".join(f"F{f} {s[f'acc_F{f}']:.3f} (+{s[f'rescue_F{f}']}/-{s[f'harm_F{f}']}, {s[f'cost_F{f}']:.1f}x)"
                    for f in (2, 4, 6, 8) if f"acc_F{f}" in s)
    print(f"- **{m}**: greedy {s['acc_greedy']:.3f}; {row}; oracle {s['acc_oracle']:.3f}")
