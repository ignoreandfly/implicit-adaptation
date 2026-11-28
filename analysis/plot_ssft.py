"""Plots for the Self-SSFT run (README "Self-SSFT"): results (accuracy, voting, diversity, per-tag) and training.

    python -m analysis.plot_ssft --res results/ssft_self --out results/ssft_self/plots
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C = {"ssft_gold": "#2a78d6", "rft_gold_m4": "#eb6834", "rft_gold_all": "#1baf7a", "ssft_gold_random": "#eda100",
     "ssft_majority": "#e87ba4", "rft_majority_m4": "#008300", "sft": "#8a8984", "instruct": "#6250d6"}
NAME = {"ssft_gold": "Self-SSFT (gold)", "rft_gold_m4": "RFT, same traces (gold)", "rft_gold_all": "RFT, all correct (gold)",
        "ssft_gold_random": "Random tags (gold)", "ssft_majority": "Self-SSFT (label-free)",
        "rft_majority_m4": "RFT, same traces (label-free)", "sft": "SFT", "instruct": "Instruct (RL)"}
GREEDY = {"ssft_gold": "ssft_gold_t1", "rft_gold_m4": "rft_gold_m4", "rft_gold_all": "rft_gold_all",
          "ssft_gold_random": "ssft_gold_random_t1", "ssft_majority": "ssft_majority_t1",
          "rft_majority_m4": "rft_majority_m4", "sft": "olmo2-1b-sft", "instruct": "olmo2-1b-instruct"}


def style(ax, title):
    ax.set_facecolor(SURF)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def enlarge(fig, k):
    """Scale every text and line in a figure so it stays readable when shrunk to a 5.5in column."""
    import matplotlib.text, matplotlib.lines
    for t in fig.findobj(matplotlib.text.Text):
        t.set_fontsize(t.get_fontsize() * k)
    for l in fig.findobj(matplotlib.lines.Line2D):
        l.set_linewidth(l.get_linewidth() * 1.3); l.set_markersize(l.get_markersize() * 1.2)


def hbar(ax, names, vals, fmt="{:.1f}"):
    y = range(len(names))
    ax.barh(list(y), vals, color=[C[n] for n in names], height=0.62, edgecolor=SURF, linewidth=2)
    ax.set_yticks(list(y)); ax.set_yticklabels([NAME[n] for n in names], fontsize=9, color=INK)
    ax.invert_yaxis(); ax.grid(axis="x", color=GRID, linewidth=0.8); ax.grid(axis="y", visible=False)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, i, fmt.format(v), va="center", fontsize=9, color=INK2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default="results/ssft_self")
    ap.add_argument("--out", default="results/ssft_self/plots")
    args = ap.parse_args()
    R, out = Path(args.res), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cons = {**json.load(open(R / "cons_gold.json"))["systems"], **json.load(open(R / "cons_majority.json"))["systems"]}
    greedy = {n: 100 * json.load(open(R / "eval" / d / "test.metrics.json"))["acc_lenient"] for n, d in GREEDY.items()}
    order = ["ssft_gold", "rft_gold_m4", "rft_gold_all", "ssft_gold_random", "ssft_majority", "rft_majority_m4",
             "sft", "instruct"]

    # ---------- results ----------
    fig, axs = plt.subplots(2, 2, figsize=(13, 9), facecolor=SURF)
    ax = axs[0, 0]; style(ax, "One answer per problem (greedy; tagged models use <think1>)")
    hbar(ax, order, [greedy[n] for n in order]); ax.set_xlim(50, 70); ax.set_xlabel("test500 accuracy (%)", color=INK2)

    ax = axs[0, 1]; style(ax, "Majority vote over 4 paths (equal compute)")
    hbar(ax, order, [100 * cons[n]["4"]["vote"] for n in order]); ax.set_xlim(55, 78)
    ax.set_xlabel("test500 vote@4 accuracy (%)", color=INK2)

    ax = axs[1, 0]; style(ax, "Vote@k: tags (one greedy path each) vs temperature samples")
    for n in ["ssft_gold", "rft_gold_m4", "rft_gold_all", "ssft_gold_random", "sft"]:
        ks = [1, 2, 3, 4]; v = [100 * cons[n][str(k)]["vote"] for k in ks]
        ax.plot(ks, v, color=C[n], linewidth=2, marker="o", markersize=7, markeredgecolor=SURF, markeredgewidth=2,
                label=NAME[n])
    ax.set_xticks([1, 2, 3, 4]); ax.set_xlabel("paths per problem (k)", color=INK2); ax.set_ylabel("vote@k (%)", color=INK2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)

    ax = axs[1, 1]; style(ax, "Diversity: distinct answers among 4 paths")
    names = ["ssft_gold", "rft_gold_m4", "rft_gold_all", "ssft_gold_random", "ssft_majority", "rft_majority_m4", "sft", "instruct"]
    hbar(ax, names, [cons[n]["4"]["distinct_answers"] for n in names], "{:.2f}"); ax.set_xlim(1, 2.7)
    ax.set_xlabel("mean distinct answers per problem (1 = all paths agree)", color=INK2)
    fig.suptitle("Self-SSFT vs RFT on OLMo 2 1B SFT - GSM8K test500 (500 problems; +/-2.2 pts noise)", x=0.01, ha="left",
                 fontsize=13, color=INK, fontweight="bold")
    enlarge(fig, 1.3); fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(out / "ssft_results.png", dpi=150, facecolor=SURF); plt.close(fig)

    # ---------- per-tag accuracy ----------
    fig, ax = plt.subplots(figsize=(8, 4.2), facecolor=SURF); style(ax, "Accuracy of each tag (greedy)")
    tags = ["<think1>", "<think2>", "<think3>", "<think4>"]; w = 0.26
    for i, n in enumerate(["ssft_gold", "ssft_majority", "ssft_gold_random"]):
        v = [100 * a for a in cons[n]["per_path_acc"]]
        ax.bar([t + (i - 1) * w for t in range(4)], v, width=w, color=C[n], label=NAME[n], edgecolor=SURF, linewidth=2)
    ax.axhline(greedy["rft_gold_m4"], color=C["rft_gold_m4"], linestyle="--", linewidth=1.5, label="RFT, same traces (greedy)")
    ax.set_xticks(range(4)); ax.set_xticklabels(tags, color=INK); ax.set_ylim(50, 66); ax.set_ylabel("test500 accuracy (%)", color=INK2)
    ax.legend(frameon=False, fontsize=9, ncol=2, loc="upper right")
    fig.tight_layout(); fig.savefig(out / "ssft_per_tag.png", dpi=150, facecolor=SURF); plt.close(fig)

    # ---------- training ----------
    logs = {n: json.load(open(R / "configs" / f"{n}.json"))["log"]
            for n in ["ssft_gold", "ssft_majority", "ssft_gold_random", "rft_gold_m4"]}
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.6), facecolor=SURF)
    ax = axs[0]; style(ax, "Training loss (one batch/point)")
    for n, L in logs.items():
        ax.plot([r["step"] for r in L], [r["loss"] for r in L], color=C[n], linewidth=2, label=NAME[n])
    ax.set_xlabel("optimizer step", color=INK2); ax.set_ylabel("token NLL", color=INK2)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))

    ax = axs[1]; style(ax, "Same assignment as epoch 1")
    for n in ["ssft_gold", "ssft_majority"]:
        pts = [(r["step"], 100 * r["assignment_stable"]) for r in logs[n] if "assignment_stable" in r]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=C[n], linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURF, markeredgewidth=2, label=NAME[n])
    ax.axhline(100 / 24, color=INK2, linestyle=":", linewidth=1.2); ax.text(ax.get_xlim()[0], 7, " chance ~4% (1 of 24 assignments)", fontsize=8, color=INK2)
    ax.set_ylim(0, 100); ax.set_xlabel("optimizer step (epoch 2)", color=INK2); ax.set_ylabel("% of questions", color=INK2)
    ax.legend(frameon=False, fontsize=9)

    ax = axs[2]; style(ax, "Tag usage, Self-SSFT (gold)")
    L = logs["ssft_gold"]; steps = [r["step"] for r in L]; bottom = [0] * len(L)
    shades = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
    for t in range(4):
        share = [100 * r["tag_usage"].get(str(t), 0) / sum(r["tag_usage"].values()) for r in L]
        ax.bar(steps, share, bottom=bottom, width=16, color=shades[t], edgecolor=SURF, linewidth=1, label=tags[t])
        bottom = [b + s for b, s in zip(bottom, share)]
    ax.axhline(25, color=INK, linewidth=0.8, linestyle=":"); ax.set_ylim(0, 100)
    ax.set_xlabel("optimizer step", color=INK2); ax.set_ylabel("% of matched traces", color=INK2)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    enlarge(fig, 1.45); fig.tight_layout(); fig.savefig(out / "ssft_training.png", dpi=150, facecolor=SURF); plt.close(fig)
    print("wrote", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
