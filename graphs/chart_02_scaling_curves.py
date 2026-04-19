"""Chart 2 — Scaling curves: output_throughput vs concurrency, 3 panels × 5 models."""
import matplotlib.pyplot as plt
from pathlib import Path

from data_loader import load_595, series
from theme import (apply_theme, title_block, COLORS, MODEL_DISPLAY, MODEL_ORDER,
                   INK, MUTED, PROFILE_DESC)

OUT = Path(__file__).parent / "output" / "chart_02_scaling_curves.png"


def main():
    apply_theme()
    rows = load_595()

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.4), sharey=False)
    fig.subplots_adjust(left=0.055, right=0.98, top=0.80, bottom=0.13, wspace=0.23)

    for ax, profile in zip(axes, ["1k1k", "1k4k", "4k1k"]):
        peak_y = 0
        for m in MODEL_ORDER:
            pts = series(rows, m, profile, "output_throughput")
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=COLORS[m], marker="o",
                    markerfacecolor="white", markeredgewidth=1.8,
                    label=MODEL_DISPLAY[m])
            ix = ys.index(max(ys))
            ax.scatter([xs[ix]], [ys[ix]], s=95, color=COLORS[m],
                       edgecolor="white", linewidth=1.6, zorder=5)
            ax.annotate(f"{ys[ix]/1000:.1f}k @ c={xs[ix]}",
                        xy=(xs[ix], ys[ix]),
                        xytext=(6, 7), textcoords="offset points",
                        fontsize=8, color=COLORS[m], fontweight="bold")
            peak_y = max(peak_y, max(ys))

        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
        ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "128", "256", "512"])
        ax.set_title(f"{profile}  ·  {PROFILE_DESC[profile]}",
                     loc="left", color=INK, pad=8)
        ax.set_xlabel("Concurrent requests", color=MUTED)
        if ax is axes[0]:
            ax.set_ylabel("Output throughput (tok/s)", color=MUTED)
        ax.set_ylim(0, peak_y * 1.15)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))

    axes[0].legend(loc="upper left", fontsize=9, ncol=1,
                   handlelength=1.3, handletextpad=0.6, borderpad=0.2)

    title_block(
        fig,
        "Scaling behavior: where each model saturates B300",
        "Output throughput vs. concurrent requests across three sequence-length profiles. Markers = sweep points; filled dot = peak.",
    )

    OUT.parent.mkdir(exist_ok=True)
    plt.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
