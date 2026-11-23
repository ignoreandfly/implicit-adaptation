"""Milestone figure: (a) voting vs any-of-k accuracy across stages (500 test problems, 64 samples, T 0.7, averaged over
100 random sample orders; vote ties split evenly); (b) share of the SFT-to-Instruct gap closed at 1x cost (full test, fp32
contrast run), paired bootstrap 95% intervals over problems.   python -m analysis.plot_milestone"""
import json
from collections import Counter
from math import comb
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#898781", "text.color": "#0b0b0b", "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
OUT = Path("results/figures"); OUT.mkdir(parents=True, exist_ok=True)
KS = [1, 2, 4, 8, 16, 32, 64]; COL = {"sft": "#2a78d6", "dpo": "#1baf7a", "instruct": "#eb6834"}; NAME = {"sft": "SFT", "dpo": "DPO", "instruct": "Instruct (RL)"}
rng = np.random.default_rng(0); res = {"a": {}, "b": {}}
fig, (ax, bx) = plt.subplots(1, 2, figsize=(5.5, 2.15), gridspec_kw={"width_ratios": [1, 1.05]})
for m in ("sft", "dpo", "instruct"):
    R = [json.loads(l) for l in open(f"results/olmo2-1b-{m}/gsm8k/0shot_sub500_n64_t0.7/test.generations.jsonl")]
    assert len(R) == 500 and all(len(r["scores"]) == 64 for r in R)
    A = [[s["lenient_answer"] for s in r["scores"]] for r in R]; C = [[bool(s["lenient_correct"]) for s in r["scores"]] for r in R]
    P = [np.mean([1 - comb(64 - sum(c), k) / comb(64, k) for c in C]) for k in KS]; V = []
    for k in KS:
        tot = 0.0; n = 1 if k == 64 else 100
        for _ in range(n):
            for a, c in zip(A, C):
                ix = rng.permutation(64)[:k]; v = Counter(a[i] for i in ix if a[i] is not None)
                if v:
                    top = max(v.values()); T = [x for x in v if v[x] == top]; good = {a[i] for i in ix if c[i]}
                    tot += sum(x in good for x in T) / len(T)
        V.append(tot / (n * 500))
    res["a"][m] = {"k": KS, "any": P, "vote": V}
    ax.plot(KS, 100 * np.array(V), "-o", c=COL[m], lw=1.6, ms=3.2, label=NAME[m]); ax.plot(KS, 100 * np.array(P), "--", c=COL[m], lw=1.2)
ax.set_xscale("log", base=2); ax.set_xticks(KS); ax.set_xticklabels(KS); ax.set_ylim(45, 100); ax.grid(axis="y", c="#e1e0d9", lw=.6); ax.set_axisbelow(True)
ax.set_xlabel("Samples per problem"); ax.set_ylabel("Accuracy (%)"); ax.set_title("(a) The ability is largely there", fontsize=8.5, loc="left")
ax.text(0.95, 93, "any sample correct\n(dashed)", fontsize=6.8, color="#52514e", va="center"); ax.text(66, 64.5, "majority vote (solid)", fontsize=6.8, color="#52514e", ha="right", va="center")
ax.legend(frameon=False, fontsize=6.6, loc="lower right", handlelength=1.4, borderaxespad=0, labelspacing=.25)
def L(n): return np.array([r["paths"][r["chosen"]]["score"]["lenient_correct"] for r in map(json.loads, open(f"results/contrast/{n}/test/test.generations.jsonl"))], float)
X = {n: L(n) for n in ("sft", "rft_a1", "rftcon_l1", "con_l2", "rftrand_l0.25", "instruct")}; assert all(len(x) == 1319 for x in X.values())
I = [rng.integers(0, 1319, 1319) for _ in range(5000)]
B = [("rftrand_l0.25", "RFT + random\nsplit (control)", "#b5b3ab"), ("rft_a1", "RFT", "#b5b3ab"), ("con_l2", "Correct − wrong\nonly", "#86b6ef"), ("rftcon_l1", "RFT + (correct\n− wrong)", "#2a78d6")]
for i, (n, lab, c) in enumerate(B):
    f = lambda j: 100 * (X[n][j].mean() - X["sft"][j].mean()) / (X["instruct"][j].mean() - X["sft"][j].mean())
    g = f(slice(None)); lo, hi = np.percentile([f(j) for j in I], [2.5, 97.5]); res["b"][n] = {"acc": X[n].mean(), "gap": g, "ci": [lo, hi]}
    bx.barh(i, g, .55, color=c); bx.plot([lo, hi], [i, i], c="#0b0b0b", lw=.9); bx.text(max(hi, g) + 2.5, i, f"{g:.0f}%", va="center", fontsize=7.5)
bx.axvline(100, c=COL["instruct"], lw=1.4); bx.text(99, 3.62, "Instruct (RL)", ha="right", fontsize=6.8, color="#52514e")
bx.set_yticks(range(4)); bx.set_yticklabels([b[1] for b in B], fontsize=7); bx.set_xlim(0, 112); bx.set_ylim(-.5, 3.95); bx.set_xticks([0, 25, 50, 75, 100])
bx.set_xlabel("SFT-to-Instruct gap closed (%)"); bx.set_title("(b) Recovering it at 1× cost", fontsize=8.5, loc="left"); bx.tick_params(axis="y", length=0)
fig.tight_layout(w_pad=1.2); fig.savefig(OUT / "milestone.pdf"); fig.savefig(OUT / "milestone.png", dpi=250); json.dump(res, open(OUT / "milestone.json", "w"), indent=1, default=float)
print(json.dumps({"a": {m: {k: [round(100 * x, 1) for x in v] for k, v in d.items() if k != "k"} for m, d in res["a"].items()}, "b": res["b"]}, default=float))
