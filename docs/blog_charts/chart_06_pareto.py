"""Chart 6 — Pareto frontier: per-GPU throughput vs interactivity (1k1k)."""
from pathlib import Path
import matplotlib.pyplot as plt

from _data import load_595, series
from theme_dark import (apply_dark, title_block, MODEL_COLORS, MODEL_DISPLAY,
                         MODEL_ORDER, TEXT, MUTED)

OUT = Path(__file__).parent / "pareto_frontier_1k1k.png"
N_GPUS = 8
PROFILE = "1k1k"


def main():
    apply_dark()
    rows = load_595()

    fig, ax = plt.subplots(figsize=(8, 5.3))
    fig.subplots_adjust(left=0.09, right=0.97, top=0.80, bottom=0.12)

    label_offsets = {
        "deepseek-r1":       (12, 0),
        "qwen3.5-397b-a17b": (10, 10),
        "minimax-m2.7":      (12, -4),
        "glm-5.1":           (8, -18),
        "kimi-k2.5":         (12, 4),
    }

    for m in MODEL_ORDER:
        tp = series(rows, m, PROFILE, "output_throughput")
        tt = series(rows, m, PROFILE, "mean_tpot_ms")
        tp_by_c = dict(tp)
        tt_by_c = dict(tt)
        concs = sorted(set(tp_by_c) & set(tt_by_c))
        if not concs:
            continue

        xs = [1000.0 / tt_by_c[c] for c in concs]
        ys = [tp_by_c[c] / N_GPUS for c in concs]

        ax.plot(xs, ys, color=MODEL_COLORS[m], marker="o",
                markerfacecolor=MODEL_COLORS[m],
                markeredgecolor="#1a1a2e", markeredgewidth=1.0,
                alpha=0.92)

        iy = ys.index(max(ys))
        ax.scatter([xs[iy]], [ys[iy]], s=140, color=MODEL_COLORS[m],
                   edgecolor="#E0E0E0", linewidth=1.6, zorder=5)

        dx, dy = label_offsets.get(m, (10, 0))
        ax.annotate(MODEL_DISPLAY[m], xy=(xs[iy], ys[iy]),
                    xytext=(dx, dy), textcoords="offset points",
                    fontsize=9.5, color=MODEL_COLORS[m],
                    fontweight="bold", ha="left", va="center")

    ax.set_xlabel("Interactivity (tok/s per user)", color=MUTED)
    ax.set_ylabel("Throughput per GPU (tok/s)", color=MUTED)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1000:.1f}k"))

    title_block(
        fig,
        "Pareto Frontier — Throughput per GPU vs Interactivity",
        "1k1k profile, B300 NVL8, NVFP4, SGLang. Filled outline = peak tok/s/GPU for each model.",
    )

    plt.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
