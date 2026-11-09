"""Redraws the report figures as vector PDFs.

Every value below is copied from the outputs of swarm_unmasking.ipynb (run of 7 November 2026,
AI Village main-village, 52 agents, 3 seeds). Cell names refer to that notebook.
Usage:  python make_figures.py   ->  figures/*.pdf (for the report) and figures/*.png (for the README)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"          # categorical slots 1-3, fixed order
CHANCE = "#b9b8b0"
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e6e5e0"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.titleweight": "bold",
    "axes.labelsize": 7.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.linewidth": 0.6,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
    "lines.linewidth": 1.5, "pdf.fonttype": 42,
})
pct = lambda v: f"{v * 100:.0f}%"


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=220, facecolor="white")
    plt.close(fig)


def robustness():
    # Cells "Experiment 1" (standard, all text), "Experiment 6" (6a time gap) and "Experiment 7" (7a hard mode)
    labels = ["All rivals,\nany time", ">1 day", ">7 days", ">30 days", ">1 day", ">7 days"]
    top1 = [0.833, 0.808, 0.741, 0.611, 0.747, 0.631]
    chance = [0.021, 0.020, 0.018, 0.015, 0.094, 0.069]
    x = np.arange(len(labels)); w = 0.36
    fig, ax = plt.subplots(figsize=(5.5, 2.15))
    ax.axvspan(3.5, 5.6, color="#f3f2ee", zorder=0, lw=0)
    ax.bar(x - w / 2 - 0.01, top1, w, color=BLUE, label="Top-1 accuracy (all text features)", zorder=3, edgecolor="white", lw=0.8)
    ax.bar(x + w / 2 + 0.01, chance, w, color=CHANCE, label="Chance", zorder=3, edgecolor="white", lw=0.8)
    for i, (v, c) in enumerate(zip(top1, chance)):
        ax.text(i - w / 2 - 0.01, v + 0.02, pct(v), ha="center", va="bottom", fontsize=7, color=INK)
        ax.text(i + w / 2 + 0.01, c + 0.02, pct(c), ha="center", va="bottom", fontsize=6.5, color=MUTED)
    ax.text(0, 0.95, "Standard", ha="center", fontsize=7, color=MUTED, fontweight="bold")
    ax.text(2, 0.91, "Time-gap test:\nevery candidate this far away", ha="center", fontsize=7, color=MUTED, fontweight="bold", linespacing=1.1)
    ax.text(4.5, 0.91, "Hard mode: same-month rivals,\ntrue match this far away", ha="center", fontsize=7, color=MUTED, fontweight="bold", linespacing=1.1)
    ax.axvline(0.5, color=GRID, lw=0.8, zorder=1)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlabel("Minimum time gap")
    ax.set_ylim(0, 1.03); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlim(-0.6, 5.6); ax.grid(axis="x", visible=False)
    ax.set_ylabel("Top-1 accuracy")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    fig.tight_layout(pad=0.3)
    save(fig, "fig_robustness")


def data_and_drift():
    # Cell "Experiment 2" (messages per alias) and "Experiment 7" (7g drift, all text features)
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.5, 1.9))
    xs = [5, 10, 20, 40]
    series = [("All features", BLUE, [0.608, 0.693, 0.739, 0.782]),
              ("Style only", ORANGE, [0.608, 0.710, 0.789, 0.826]),
              ("Character n-grams", AQUA, [0.728, 0.733, 0.718, 0.691])]
    for name, col, ys in series:
        a.plot(xs, ys, marker="o", ms=4.5, color=col, label=name, mec="white", mew=0.6, zorder=3)
    a.plot(xs, [0.019, 0.019, 0.021, 0.024], ls="--", color=MUTED, lw=1, label="Chance")
    a.set_xscale("log", base=2); a.set_xticks(xs); a.set_xticklabels(xs); a.minorticks_off()
    a.set_ylim(0, 1); a.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); a.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    a.set_xlabel("Messages per alias"); a.set_ylabel("Top-1 accuracy")
    a.set_title("(a) Data per alias", loc="left")
    a.legend(loc="lower right", fontsize=6.5, handlelength=1.6)

    bins = ["<1 d", "1–7 d", "7–30 d", "30–90 d", ">90 d"]
    same = [0.480, 0.393, 0.315, 0.248, 0.170]
    diff = [0.123, 0.114, 0.100, 0.087, 0.046]
    xb = np.arange(len(bins))
    b.plot(xb, same, marker="o", ms=4.5, color=BLUE, label="Same agent", mec="white", mew=0.6, zorder=3)
    b.plot(xb, diff, marker="o", ms=4.5, color=ORANGE, label="Different agents", mec="white", mew=0.6, zorder=3)
    for i in (0, len(bins) - 1):
        b.text(xb[i], same[i] + 0.03, f"{same[i]:.2f}", ha="center", fontsize=6.5)
        b.text(xb[i], diff[i] + 0.03, f"{diff[i]:.2f}", ha="center", fontsize=6.5, color=MUTED)
    b.set_xticks(xb); b.set_xticklabels(bins); b.set_ylim(0, 0.6)
    b.set_xlabel("Time between the two aliases"); b.set_ylabel("Mean cosine similarity")
    b.set_title("(b) Fingerprint drift", loc="left")
    b.legend(loc="upper right", fontsize=6.5, handlelength=1.6)
    fig.tight_layout(pad=0.3)
    save(fig, "fig_data_drift")


def disguise():
    # Cell "Experiment 3": disguised queries matched against clean references (mean ± sd over 3 seeds)
    cats = ["No disguise", "Strip\nformatting", "Reword", "Shorten", "All three"]
    allf = [0.709, 0.606, 0.708, 0.536, 0.433]; allf_sd = [0.010, 0.026, 0.011, 0.014, 0.012]
    sty = [0.744, 0.530, 0.735, 0.424, 0.267]; sty_sd = [0.015, 0.032, 0.010, 0.025, 0.008]
    x = np.arange(len(cats)); w = 0.36
    fig, ax = plt.subplots(figsize=(5.5, 1.75))
    ax.bar(x - w / 2 - 0.01, allf, w, yerr=allf_sd, color=BLUE, label="All features", zorder=3, edgecolor="white", lw=0.8,
           error_kw={"ecolor": MUTED, "elinewidth": 0.8, "capsize": 1.5})
    ax.bar(x + w / 2 + 0.01, sty, w, yerr=sty_sd, color=ORANGE, label="Style only", zorder=3, edgecolor="white", lw=0.8,
           error_kw={"ecolor": MUTED, "elinewidth": 0.8, "capsize": 1.5})
    for i in range(len(cats)):
        ax.text(i - w / 2 - 0.01, allf[i] + 0.035, pct(allf[i]), ha="center", fontsize=6.6)
        ax.text(i + w / 2 + 0.01, sty[i] + 0.035, pct(sty[i]), ha="center", fontsize=6.6)
    ax.axhline(0.022, color=MUTED, ls="--", lw=1, zorder=2, label="Chance (2%)")
    ax.set_xticks(x); ax.set_xticklabels(cats)
    ax.set_ylim(0, 0.97); ax.set_yticks([0, 0.25, 0.5, 0.75]); ax.set_yticklabels(["0", "25%", "50%", "75%"])
    ax.set_ylabel("Top-1 accuracy"); ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncol=3)
    fig.tight_layout(pad=0.3)
    save(fig, "fig_disguise")


if __name__ == "__main__":
    robustness(); data_and_drift(); disguise()
    print("wrote", sorted(os.listdir(OUT)))
