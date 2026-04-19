"""Chart 3 — Driver 590.48 vs 595 grouped bars, sorted by delta."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from theme_dark import (apply_dark, title_block, MODEL_COLORS, MODEL_DISPLAY,
                         TEXT, MUTED, POSITIVE, NEGATIVE)

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
    apply_dark()

    labels = [MODEL_DISPLAY[m] for m, _, _ in DATA]
    v590   = [a for _, a, _ in DATA]
    v595   = [b for _, _, b in DATA]

    fig, ax = plt.subplots(figsize=(8.5, 4.3))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.78, bottom=0.16)

    x = np.arange(len(labels))
    w = 0.37
    ax.bar(x - w / 2, v590, w, label="Driver 590.48",
           color="#4B5563", edgecolor="none")
    ax.bar(x + w / 2, v595, w, label="Driver 595",
           color=[MODEL_COLORS[m] for m, _, _ in DATA], edgecolor="none")

    ymax = max(v595 + v590)
    for i, (m, a, b) in enumerate(DATA):
        delta = (b - a) / a * 100
        col = POSITIVE if delta >= 0 else NEGATIVE
        sign = "+" if delta >= 0 else ""
        ax.text(i, max(a, b) + ymax * 0.045,
                f"{sign}{delta:.1f}%",
                ha="center", fontsize=11, fontweight="bold", color=col)
        ax.text(i + w / 2, b + ymax * 0.008, f"{b:,}",
                ha="center", va="bottom", fontsize=8.5, color=TEXT)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=TEXT, fontsize=9.5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    ax.set_ylabel("Peak output tok/s", color=MUTED)
    ax.set_ylim(0, ymax * 1.20)
    ax.legend(loc="upper right")
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
