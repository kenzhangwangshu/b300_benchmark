"""Chart 2 — Three-panel scaling curves, output tok/s vs concurrency."""
from pathlib import Path
import matplotlib.pyplot as plt

from _data import load_595, series
from theme_dark import (apply_dark, title_block, MODEL_COLORS, MODEL_DISPLAY,
                         MODEL_ORDER, TEXT, MUTED, GRID)

OUT = Path(__file__).parent / "scaling_curves_all_profiles.png"

PANEL_TITLES = {
    "1k1k": "1k1k  —  Balanced",
    "1k4k": "1k4k  —  Decode-Heavy",
    "4k1k": "4k1k  —  Prefill-Heavy",
}


def main():
    apply_dark()
    rows = load_595()

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=False)
    fig.subplots_adjust(left=0.05, right=0.985, top=0.78, bottom=0.22, wspace=0.22)

    handles, labels = [], []

    for ax, profile in zip(axes, ["1k1k", "1k4k", "4k1k"]):
        peak_y = 0
        for m in MODEL_ORDER:
            pts = series(rows, m, profile, "output_throughput")
            if not pts:
                continue
            xs, ys = zip(*pts)
            (line,) = ax.plot(xs, ys, color=MODEL_COLORS[m],
                              marker="o", markerfacecolor=MODEL_COLORS[m],
                              markeredgecolor="#1a1a2e", markeredgewidth=1.2,
                              label=MODEL_DISPLAY[m])
            ix = ys.index(max(ys))
            knee_x = xs[ix]
            ax.axvline(knee_x, color=MODEL_COLORS[m], linestyle="--",
                       linewidth=0.8, alpha=0.35, zorder=1)
            ax.scatter([knee_x], [ys[ix]], s=110, color=MODEL_COLORS[m],
                       edgecolor="#1a1a2e", linewidth=1.5, zorder=5)
            ax.annotate(f"{ys[ix]/1000:.1f}k",
                        xy=(knee_x, ys[ix]),
                        xytext=(8, 7), textcoords="offset points",
                        fontsize=8.5, color=MODEL_COLORS[m],
                        fontweight="bold")
            if ax is axes[0]:
                handles.append(line)
                labels.append(MODEL_DISPLAY[m])
            peak_y = max(peak_y, max(ys))

        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
        ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "128", "256", "512"])
        ax.set_title(PANEL_TITLES[profile], loc="left", color=TEXT, pad=8)
        ax.set_xlabel("Concurrency", color=MUTED)
        if ax is axes[0]:
            ax.set_ylabel("Output tok/s", color=MUTED)
        ax.set_ylim(0, peak_y * 1.15)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
        ax.grid(color=GRID, linewidth=0.4)

    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, 0.05), ncol=5,
               handlelength=1.5, handletextpad=0.55, columnspacing=2.0,
               labelcolor=TEXT, fontsize=10)

    title_block(
        fig,
        "Scaling behavior: where each model saturates B300",
        "Output tok/s vs. concurrency, three sequence-length profiles. Filled dot = peak; dashed line = knee concurrency.",
        footer="Source: B300 NVL8, Driver 595, SGLang 0.5.10.post1, NVFP4, TP=8",
    )

    plt.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
