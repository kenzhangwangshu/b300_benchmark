"""Chart 6 — Per-GPU throughput vs active parameters (1k1k peak). Pareto lens."""
import matplotlib.pyplot as plt
from pathlib import Path

from data_loader import load_595, peaks
from theme import (apply_theme, title_block, COLORS, MODEL_DISPLAY, MODEL_ORDER,
                   MODEL_PARAMS, INK, MUTED, FAINT)

OUT = Path(__file__).parent / "output" / "chart_06_per_gpu_efficiency.png"
PROFILE = "1k1k"
N_GPUS = 8


def main():
    apply_theme()
    pk = peaks(load_595())

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    fig.subplots_adjust(left=0.085, right=0.97, top=0.80, bottom=0.13)

    for m in MODEL_ORDER:
        r = pk[m].get(PROFILE)
        if r is None:
            continue
        active = MODEL_PARAMS[m]["active_b"]
        per_gpu = r["output_throughput"] / N_GPUS
        total   = MODEL_PARAMS[m]["total_b"]
        size = 160 + (total / 10.0)
        ax.scatter([active], [per_gpu], s=size,
                   color=COLORS[m], edgecolor="white", linewidth=2, zorder=3)
        ax.annotate(
            f"{MODEL_DISPLAY[m]}\n{per_gpu:,.0f} tok/s/GPU · {total}B total",
            xy=(active, per_gpu),
            xytext=(10, -6), textcoords="offset points",
            fontsize=9, color=INK, fontweight="bold",
        )

    ax.set_xlabel("Active parameters per token (billions)", color=MUTED)
    ax.set_ylabel("Peak output throughput per GPU (tok/s)", color=MUTED)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1000:.1f}k"))

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim(0, xlim[1] * 1.18)
    ax.set_ylim(0, ylim[1] * 1.15)

    ax.text(0.99, 0.02, "bubble size ∝ total parameters",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color=FAINT, style="italic")

    title_block(
        fig,
        "Smaller active parameters ≠ higher per-GPU throughput",
        f"Peak per-GPU output tok/s vs. active-parameter count, {PROFILE} on 8×B300. Bubble size shows model footprint.",
    )

    OUT.parent.mkdir(exist_ok=True)
    plt.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
