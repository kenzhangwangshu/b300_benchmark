"""Chart 7 — MiniMax M2.7 cluster economics across deployment shapes."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from theme import (apply_theme, title_block, INK, MUTED,
                   POSITIVE, NEGATIVE, HIGHLIGHT, FAINT)

OUT = Path(__file__).parent / "cluster_economics_m27.png"

DATA = [
    ("TP=1",        "1 GPU / instance",   768, 21700, 3100, 85.5),
    ("TP=8",        "1 node / instance",   96,  4600, 3100, 31.4),
    ("TP=8 · EP=4", "4 nodes / instance",  24,  4600, 3100, 31.4),
]


def main():
    apply_theme()

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.18)

    labels = [row[0] for row in DATA]
    revenue = [row[3] for row in DATA]
    cost    = [row[4] for row in DATA]

    x = np.arange(len(labels))
    w = 0.37
    ax.bar(x - w / 2, revenue, w, label="Revenue / hr",
           color=HIGHLIGHT, edgecolor="white", linewidth=0.8)
    ax.bar(x + w / 2, cost, w, label="Cost / hr",
           color=FAINT, edgecolor="white", linewidth=0.8)

    ymax = max(revenue + cost)
    for i, (lab, sub, reps, rev, cst, m) in enumerate(DATA):
        col = POSITIVE if m >= 50 else NEGATIVE
        ax.text(i, max(rev, cst) + ymax * 0.065,
                f"{m:.1f}% margin",
                ha="center", fontsize=11, fontweight="bold", color=col)
        ax.text(i - w / 2, rev + ymax * 0.012, f"\\${rev/1000:.1f}k",
                ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=INK)
        ax.text(i + w / 2, cst + ymax * 0.012, f"\\${cst/1000:.1f}k",
                ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=INK)

    tick_labels = [f"{lab}\n{reps} replicas · {sub}"
                   for lab, sub, reps, *_ in DATA]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, color=INK, fontsize=10)
    ax.set_ylabel("$/hr", color=MUTED)
    ax.set_ylim(0, ymax * 1.25)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"\\${v/1000:.0f}k"))
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(axis="x", visible=False)

    title_block(
        fig,
        "MiniMax M2.7 Cluster Economics: Horizontal Scaling Wins",
        r"96 nodes (768 GPUs), NVFP4, same hardware cost. Revenue assumes \$3.01/MTok output.",
    )

    plt.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
