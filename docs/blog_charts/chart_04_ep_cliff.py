"""Chart 4 — EP cliff: DeepSeek R1 with expert-parallelism enabled."""
from pathlib import Path
import matplotlib.pyplot as plt

from theme import (apply_theme, title_block, INK, MUTED,
                   POSITIVE, NEGATIVE, HIGHLIGHT)

OUT = Path(__file__).parent / "ep_cliff_deepseek_r1.png"

CONC = [1, 2, 4, 8, 16]
TPS  = [134, 258, 469, 815, 146]
TPOT = [7.1, 7.7, 8.5, 9.4, 1425]


def main():
    apply_theme()

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    fig.subplots_adjust(left=0.08, right=0.90, top=0.78, bottom=0.14)

    ax.axvspan(0.9, 8.5, alpha=0.08, color=POSITIVE, zorder=0)
    ax.axvspan(8.5, 18, alpha=0.10, color=NEGATIVE, zorder=0)

    ln1, = ax.plot(CONC, TPS, color=HIGHLIGHT, marker="o",
                   markerfacecolor="white", markeredgewidth=1.8,
                   label="Throughput (tok/s)")
    ax.set_xlabel("Concurrency", color=MUTED)
    ax.set_ylabel("Throughput (tok/s)", color=HIGHLIGHT)
    ax.tick_params(axis="y", labelcolor=HIGHLIGHT)
    ax.set_xlim(0.8, 20)
    ax.set_ylim(0, 1000)
    ax.set_xscale("log", base=2)
    ax.set_xticks(CONC)
    ax.set_xticklabels([str(c) for c in CONC])

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(NEGATIVE)
    ln2, = ax2.plot(CONC, TPOT, color=NEGATIVE, linestyle="--", marker="s",
                    markerfacecolor="white", markeredgewidth=1.8,
                    label="TPOT (ms)")
    ax2.set_ylabel("TPOT (ms, log scale)", color=NEGATIVE)
    ax2.tick_params(axis="y", labelcolor=NEGATIVE)
    ax2.set_yscale("log")
    ax2.set_ylim(5, 3000)
    ax2.grid(False)

    ax.scatter([16], [146], s=280, marker="X", color=NEGATIVE,
               edgecolor="white", linewidth=1.5, zorder=10)

    ax.annotate(
        "96/160 requests failed\nTPOT exploded to 1,425 ms",
        xy=(16, 146), xytext=(-130, 80), textcoords="offset points",
        fontsize=9, color=INK, fontweight="bold",
        ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.55", facecolor="#FDECEC",
                  edgecolor=NEGATIVE, linewidth=1),
        arrowprops=dict(arrowstyle="->", color=NEGATIVE, linewidth=1.2),
    )

    ax.text(2.8, 955, "HEALTHY ZONE", fontsize=8.5, color=POSITIVE,
            fontweight="bold", ha="center")
    ax.text(15.5, 955, "FAILURE", fontsize=8.5, color=NEGATIVE,
            fontweight="bold", ha="center")

    ax.legend(handles=[ln1, ln2], loc="upper left", fontsize=9.5)

    title_block(
        fig,
        "Expert Parallelism Cliff — DeepSeek R1 NVFP4",
        "EP works up to concurrency 8, then catastrophically fails.",
    )

    plt.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
