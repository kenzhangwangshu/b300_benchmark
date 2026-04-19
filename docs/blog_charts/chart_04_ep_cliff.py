"""Chart 4 — EP cliff: DeepSeek R1 with expert-parallelism enabled."""
from pathlib import Path
import matplotlib.pyplot as plt

from theme_dark import (apply_dark, title_block, TEXT, MUTED,
                         POSITIVE, NEGATIVE)

OUT = Path(__file__).parent / "ep_cliff_deepseek_r1.png"

CONC = [1, 2, 4, 8, 16]
TPS  = [134, 258, 469, 815, 146]
TPOT = [7.1, 7.7, 8.5, 9.4, 1425]
FAILED = [0, 0, 0, 0, 96]
TOTAL = 160


def main():
    apply_dark()

    fig, ax = plt.subplots(figsize=(7.0, 3.7))
    fig.subplots_adjust(left=0.10, right=0.88, top=0.78, bottom=0.17)

    ax.axvspan(0.9, 8.5, alpha=0.10, color=POSITIVE, zorder=0)
    ax.axvspan(8.5, 18, alpha=0.10, color=NEGATIVE, zorder=0)

    ln1, = ax.plot(CONC, TPS, color="#4ECDC4", marker="o",
                   markerfacecolor="#4ECDC4", markeredgecolor="#1a1a2e",
                   markeredgewidth=1.2, label="Throughput (tok/s)")
    ax.set_xlabel("Concurrency", color=MUTED)
    ax.set_ylabel("Throughput (tok/s)", color="#4ECDC4")
    ax.tick_params(axis="y", labelcolor="#4ECDC4")
    ax.set_xlim(0.8, 20)
    ax.set_ylim(0, 1000)
    ax.set_xscale("log", base=2)
    ax.set_xticks(CONC)
    ax.set_xticklabels([str(c) for c in CONC])

    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(MUTED)
    ln2, = ax2.plot(CONC, TPOT, color="#F87171", linestyle="--", marker="s",
                    markerfacecolor="#F87171", markeredgecolor="#1a1a2e",
                    markeredgewidth=1.2, label="TPOT (ms)")
    ax2.set_ylabel("TPOT (ms)", color="#F87171")
    ax2.tick_params(axis="y", labelcolor="#F87171")
    ax2.set_yscale("log")
    ax2.set_ylim(5, 3000)
    ax2.grid(False)

    ax.scatter([16], [146], s=280, marker="X", color="#F87171",
               edgecolor="#1a1a2e", linewidth=1.5, zorder=10)

    ax.annotate(
        "96/160 requests failed\nTPOT exploded to 1,425 ms",
        xy=(16, 146), xytext=(-115, 80), textcoords="offset points",
        fontsize=9, color=TEXT, fontweight="bold",
        ha="left", va="center",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#2A1F2F",
                  edgecolor="#F87171", linewidth=1),
        arrowprops=dict(arrowstyle="->", color="#F87171", linewidth=1.2),
    )

    ax.text(2.8, 950, "HEALTHY ZONE", fontsize=8.5, color=POSITIVE,
            fontweight="bold", alpha=0.95, ha="center")
    ax.text(15.5, 950, "FAILURE", fontsize=8.5, color=NEGATIVE,
            fontweight="bold", alpha=0.95, ha="center")

    ax.legend(handles=[ln1, ln2], loc="upper left", fontsize=9,
              labelcolor=TEXT)

    title_block(
        fig,
        "Expert Parallelism Cliff — DeepSeek R1 NVFP4",
        "EP works up to concurrency 8, then catastrophically fails.",
    )

    plt.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
