"""Chart 5 — Median TTFT vs concurrency across models (1k1k)."""
import matplotlib.pyplot as plt
from pathlib import Path

from data_loader import load_595, series
from theme import (apply_theme, title_block, COLORS, MODEL_DISPLAY, MODEL_ORDER,
                   INK, MUTED)

OUT = Path(__file__).parent / "output" / "chart_05_ttft_curves.png"
PROFILE = "1k1k"


def main():
    apply_theme()
    rows = load_595()

    fig, ax = plt.subplots(figsize=(10.2, 5.4))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.80, bottom=0.13)

    max_y = 0
    for m in MODEL_ORDER:
        pts = series(rows, m, PROFILE, "median_ttft_ms")
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=COLORS[m], marker="o",
                markerfacecolor="white", markeredgewidth=1.8,
                label=MODEL_DISPLAY[m])
        max_y = max(max_y, max(ys))

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "128", "256", "512"])
    ax.set_xlabel("Concurrent requests", color=MUTED)
    ax.set_ylabel("Median TTFT (ms, log scale)", color=MUTED)
    ax.legend(loc="upper left", fontsize=9.5, ncol=1)

    title_block(
        fig,
        "Time-to-first-token scales linearly with load",
        f"Median TTFT vs. concurrency, {PROFILE} profile. Lower is better. Log-log axes.",
    )

    OUT.parent.mkdir(exist_ok=True)
    plt.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
