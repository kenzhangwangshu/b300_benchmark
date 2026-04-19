"""Chart 3 — Driver uplift: peak 1k1k output_throughput, driver 590.48 vs 595."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from data_loader import load_595, load_590, peaks
from theme import (apply_theme, title_block, COLORS, MODEL_DISPLAY, MODEL_ORDER,
                   INK, MUTED, POSITIVE, NEGATIVE, FAINT)

OUT = Path(__file__).parent / "output" / "chart_03_driver_uplift.png"
PROFILE = "1k1k"


def main():
    apply_theme()
    pk595 = peaks(load_595())
    pk590 = peaks(load_590())

    rows = []
    for m in MODEL_ORDER:
        r595 = pk595[m].get(PROFILE)
        r590 = pk590[m].get(PROFILE)
        v595 = r595["output_throughput"] if r595 else None
        v590 = r590["output_throughput"] if r590 else None
        rows.append((m, v590, v595))

    labels = [MODEL_DISPLAY[m] for m, _, _ in rows]
    v590s  = [v if v is not None else 0 for _, v, _ in rows]
    v595s  = [v if v is not None else 0 for _, _, v in rows]

    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    fig.subplots_adjust(left=0.06, right=0.97, top=0.80, bottom=0.14)

    x = np.arange(len(labels))
    w = 0.38
    ax.bar(x - w / 2, v590s, w, label="driver 590.48",
           color=FAINT, edgecolor="white", linewidth=0.8)
    bars595 = ax.bar(x + w / 2, v595s, w, label="driver 595",
                     color=[COLORS[m] for m, _, _ in rows],
                     edgecolor="white", linewidth=0.8)

    for i, (m, v590, v595) in enumerate(rows):
        if v595:
            ax.text(i + w / 2, v595 + max(v595s) * 0.012, f"{v595:,.0f}",
                    ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color=INK)
        if v590 and v595:
            delta = (v595 - v590) / v590 * 100
            col = POSITIVE if delta >= 0 else NEGATIVE
            sign = "+" if delta >= 0 else ""
            ax.text(i, max(v590, v595) + max(v595s) * 0.065,
                    f"{sign}{delta:.1f}%", ha="center", fontsize=10.5,
                    fontweight="bold", color=col)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, color=INK)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    ax.set_ylabel("Peak output throughput (tok/s)", color=MUTED)
    ax.set_ylim(0, max(v595s + v590s) * 1.18)
    ax.legend(loc="upper right", fontsize=9.5)
    ax.grid(axis="x", visible=False)

    title_block(
        fig,
        "Driver 595 delivers the silent uplift",
        f"{PROFILE} peak output throughput per model, CUDA-13.1 driver 595 vs. 590.48 on the same B300 node. Delta in green/red.",
    )

    OUT.parent.mkdir(exist_ok=True)
    plt.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
