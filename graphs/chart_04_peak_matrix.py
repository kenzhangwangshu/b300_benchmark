"""Chart 4 — Peak matrix heatmap: model × profile output throughput."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

from data_loader import load_595, peaks, PROFILES
from theme import (apply_theme, title_block, MODEL_DISPLAY, MODEL_ORDER,
                   INK, MUTED)

OUT = Path(__file__).parent / "output" / "chart_04_peak_matrix.png"


def main():
    apply_theme()
    pk = peaks(load_595())

    m = np.full((len(MODEL_ORDER), len(PROFILES)), np.nan)
    c = [[None] * len(PROFILES) for _ in MODEL_ORDER]
    for i, mk in enumerate(MODEL_ORDER):
        for j, pf in enumerate(PROFILES):
            r = pk[mk].get(pf)
            if r is not None:
                m[i, j] = r["output_throughput"]
                c[i][j] = r["concurrency"]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    fig.subplots_adjust(left=0.21, right=0.88, top=0.80, bottom=0.10)

    cmap = LinearSegmentedColormap.from_list(
        "b300", ["#F7F2EC", "#E9A23B", "#D4574E", "#8B2A28"]
    )
    vmax = np.nanmax(m)
    im = ax.imshow(m, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")

    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            v = m[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=11, color=MUTED)
                continue
            color = "white" if v > vmax * 0.55 else INK
            ax.text(j, i - 0.08, f"{v/1000:.1f}k", ha="center", va="center",
                    fontsize=13, fontweight="bold", color=color)
            ax.text(j, i + 0.22, f"c={c[i][j]}", ha="center", va="center",
                    fontsize=8.5, color=color, alpha=0.9)

    ax.set_xticks(range(len(PROFILES)))
    ax.set_xticklabels([p for p in PROFILES], fontsize=10, color=INK, fontweight="bold")
    ax.set_yticks(range(len(MODEL_ORDER)))
    ax.set_yticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER], fontsize=10, color=INK)
    ax.tick_params(axis="both", length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.outline.set_visible(False)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    cbar.ax.tick_params(labelsize=8, colors=MUTED)
    cbar.set_label("tok/s", color=MUTED, fontsize=9)

    title_block(
        fig,
        "Peak output throughput: 5 models × 3 sequence profiles",
        "Cell value = peak output tok/s; subscript = concurrency at peak. Empty = sweep incomplete.",
    )

    OUT.parent.mkdir(exist_ok=True)
    plt.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
