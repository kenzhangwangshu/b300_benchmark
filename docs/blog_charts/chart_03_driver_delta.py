"""Chart 3 — Driver 590.48 vs 595 grouped bars, sorted by delta."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from theme import (apply_theme, title_block, COLORS, MODEL_DISPLAY,
                   INK, MUTED, POSITIVE, NEGATIVE, FAINT)

OUT = Path(__file__).parent / "driver_590_vs_595.png"

DATA = [
    ("deepseek-r1",        9891, 12518),
    ("qwen3.5-397b-a17b", 10652, 11124),
    ("glm-5.1",            8913,  8953),
    ("minimax-m2.7",      10284,  9710),
    ("kimi-k2.5",          2595,  2523),
]
DATA.sort(key=lambda r: (r[2] - r[1]) / r[1], reverse=True)


def main():
    apply_theme()

    labels = [MODEL_DISPLAY[m] for m, _, _ in DATA]
    v590   = [a for _, a, _ in DATA]
    v595   = [b for _, _, b in DATA]

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.80, bottom=0.14)

    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, v590, w, label="Driver 590.48",
           color=FAINT, edgecolor="white", linewidth=0.8)
    ax.bar(x + w / 2, v595, w, label="Driver 595",
           color=[COLORS[m] for m, _, _ in DATA],
           edgecolor="white", linewidth=0.8)

    ymax = max(v595 + v590)
    for i, (m, a, b) in enumerate(DATA):
        delta = (b - a) / a * 100
        col = POSITIVE if delta >= 0 else NEGATIVE
        sign = "+" if delta >= 0 else ""
        ax.text(i, max(a, b) + ymax * 0.065, f"{sign}{delta:.1f}%",
                ha="center", fontsize=10.5, fontweight="bold", color=col)
        ax.text(i + w / 2, b + ymax * 0.012, f"{b:,}",
                ha="center", va="bottom", fontsize=9,
                fontweight="bold", color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, color=INK)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    ax.set_ylabel("Peak output throughput (tok/s)", color=MUTED)
    ax.set_ylim(0, ymax * 1.20)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(axis="x", visible=False)

    title_block(
        fig,
        "Driver 595 Impact: Not All Models Benefit",
        "Peak 1k1k throughput (tok/s), same hardware, same framework.",
    )

    plt.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
