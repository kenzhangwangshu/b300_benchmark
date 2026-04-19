"""Chart 5 — Self-hosted $/MTok vs public API reference."""
from pathlib import Path
import matplotlib.pyplot as plt

from theme import (apply_theme, title_block, MODEL_DISPLAY,
                   INK, MUTED, POSITIVE, NEGATIVE)

OUT = Path(__file__).parent / "cost_per_mtok.png"
REF_PRICE = 2.19
REF_LABEL = "OpenRouter R1 API price"

DATA = [
    ("deepseek-r1",       0.73),
    ("qwen3.5-397b-a17b", 0.82),
    ("minimax-m2.7",      0.94),
    ("glm-5.1",           1.02),
    ("kimi-k2.5",         3.61),
]
DATA.sort(key=lambda r: r[1])


def main():
    apply_theme()

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    fig.subplots_adjust(left=0.18, right=0.95, top=0.78, bottom=0.14)

    labels = [MODEL_DISPLAY[m] for m, _ in DATA]
    vals   = [v for _, v in DATA]
    colors = [POSITIVE if v < REF_PRICE else NEGATIVE for v in vals]

    y = list(range(len(labels)))[::-1]
    bars = ax.barh(y, vals, color=colors, edgecolor="white",
                   linewidth=0.8, height=0.62)

    vmax = max(max(vals), REF_PRICE)
    for bar, v in zip(bars, vals):
        ax.text(v + vmax * 0.015, bar.get_y() + bar.get_height() / 2,
                f"\\${v:.2f}", va="center", ha="left",
                fontsize=10.5, fontweight="bold", color=INK)

    ax.axvline(REF_PRICE, color=NEGATIVE, linestyle="--", linewidth=1.3)
    ax.text(REF_PRICE + 0.04, len(labels) - 0.4,
            f"{REF_LABEL} (\\${REF_PRICE:.2f})",
            color=NEGATIVE, fontsize=9, fontweight="bold", va="center")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=INK, fontsize=10, fontweight="bold")
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, vmax * 1.25)
    ax.set_xlabel("$ per million output tokens", color=MUTED)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"\\${v:.2f}"))
    ax.grid(axis="y", visible=False)

    title_block(
        fig,
        "Self-Hosted B300 Cost vs Public API",
        r"\$ per million output tokens at peak throughput, \$4.10/GPU-hr.",
    )

    plt.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
