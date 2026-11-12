"""Draw the paper figures from the CSV files written by evaluate.py.

Figures:
    fig1_research_questions.png : per-class recall for RQ1 (class imbalance) and RQ2 (label noise)
    fig2_style_accuracy.png     : test accuracy per art style for every model (mean over runs)

Usage (run from the repository root, after scripts/evaluate.py):
    python scripts/make_figures.py
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # Save figures to files without opening a window.
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Colors: one fixed color per class, used in every figure.
COLOR = {"human": "#2a78d6", "ai": "#eb6834"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

RQ1 = [("M0", "1:1"), ("M1", "1:5"), ("M2", "1:10")]        # human : ai in the train set
RQ2 = [("M0", 0), ("M3", 10), ("M4", 20), ("M5", 40)]        # % of flipped train labels
STYLES = ["art_nouveau", "baroque", "expressionism", "impressionism", "post_impressionism",
          "realism", "renaissance", "romanticism", "surrealism", "ukiyo_e"]


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def style_axes(ax):
    """Recessive axes: light grid behind the data, no top/right frame."""
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))


def plot_rq1(ax, summary):
    x = np.arange(len(RQ1))
    width = 0.34
    for i, cls in enumerate(["human", "ai"]):
        means = [summary[m][f"recall_{cls}_mean"] for m, _ in RQ1]
        stds = [summary[m][f"recall_{cls}_std"] for m, _ in RQ1]
        pos = x + (i - 0.5) * (width + 0.03)  # small gap between neighbouring bars
        ax.bar(pos, means, width, color=COLOR[cls], label=f"{cls} recall",
               yerr=stds, error_kw={"ecolor": INK_2, "elinewidth": 1, "capsize": 3})
        for p, m in zip(pos, means):
            ax.text(p, m + 0.04, f"{m:.2f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, [r for _, r in RQ1])
    ax.set_xlabel("Train ratio (human : ai)", color=INK_2, fontsize=9)
    ax.set_ylabel("Recall on test set", color=INK_2, fontsize=9)
    ax.set_title("(a) RQ1: class imbalance", loc="left", fontsize=10, color=INK)
    style_axes(ax)


def plot_rq2(ax, summary):
    x = [n for _, n in RQ2]
    for cls, marker in (("human", "o"), ("ai", "s")):
        means = np.array([summary[m][f"recall_{cls}_mean"] for m, _ in RQ2])
        stds = np.array([summary[m][f"recall_{cls}_std"] for m, _ in RQ2])
        ax.fill_between(x, means - stds, means + stds, color=COLOR[cls], alpha=0.15, linewidth=0)
        ax.plot(x, means, color=COLOR[cls], linewidth=2, marker=marker, markersize=6, label=f"{cls} recall")
    acc = [summary[m]["accuracy_mean"] for m, _ in RQ2]
    ax.plot(x, acc, color=INK_2, linewidth=1.2, linestyle="--", label="accuracy")
    ax.text(x[-1] + 1, acc[-1], f"{acc[-1]:.2f}", va="center", fontsize=8, color=INK)
    ax.text(x[0] - 1, acc[0] + 0.05, f"{acc[0]:.2f}", ha="left", fontsize=8, color=INK)
    ax.set_xticks(x, [f"{n}%" for n in x])
    ax.set_xlim(-3, 45)
    ax.set_xlabel("Flipped train labels per class", color=INK_2, fontsize=9)
    ax.set_title("(b) RQ2: label noise", loc="left", fontsize=10, color=INK)
    style_axes(ax)


def plot_style_heatmap(predictions, out_path):
    models = sorted({p["model"] for p in predictions})
    correct = defaultdict(list)
    for p in predictions:
        correct[(p["model"], p["style"])].append(p["true_label"] == p["pred_label"])
    grid = np.array([[np.mean(correct[(m, s)]) for s in STYLES] for m in models])

    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    im = ax.imshow(grid, cmap=cmap, vmin=0.5, vmax=1.0, aspect="auto")
    for i in range(len(models)):
        for j in range(len(STYLES)):
            v = grid[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v > 0.8 else INK)
    ax.set_xticks(range(len(STYLES)), [s.replace("_", " ") for s in STYLES], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(models)), models, fontsize=8)
    ax.tick_params(length=0, colors=INK_2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Accuracy (mean of 3 runs)", fontsize=8, color=INK_2)
    cbar.ax.tick_params(labelsize=7, colors=INK_2)
    cbar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results", help="Folder with the CSV files from evaluate.py")
    args = parser.parse_args()

    summary = {r["model"]: {k: float(v) for k, v in r.items() if k.endswith(("_mean", "_std"))}
               for r in read_csv(os.path.join(args.results, "metrics_summary.csv"))}
    predictions = read_csv(os.path.join(args.results, "predictions.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    plot_rq1(axes[0], summary)
    plot_rq2(axes[1], summary)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path1 = os.path.join(args.results, "fig1_research_questions.png")
    fig.savefig(path1, dpi=300)
    plt.close(fig)

    path2 = os.path.join(args.results, "fig2_style_accuracy.png")
    plot_style_heatmap(predictions, path2)
    print(f"Saved {path1}\nSaved {path2}")


if __name__ == "__main__":
    main()
