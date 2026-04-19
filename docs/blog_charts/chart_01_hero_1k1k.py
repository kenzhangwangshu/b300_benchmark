"""Chart 1 — Hero: peak 1k1k throughput, ranked."""
from pathlib import Path
import matplotlib.pyplot as plt

from _data import load_595, peaks
from theme import (apply_theme, title_block, COLORS, MODEL_DISPLAY,
                   MODEL_PARAMS, INK, MUTED)

OUT = Path(__file__).parent / "hero_throughput_1k1k.png"
PROFILE = "1k1k"


def main():
    apply_theme()
    pk = peaks(load_595())

    recs = []
    for m, profs in pk.items():
        r = profs.get(PROFILE)
        if r is None:
            continue
        recs.append((m, r["output_throughput"], r["concurrency"]))
    recs.sort(key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    fig.subplots_adjust(left=0.035, right=0.96, top=0.80, bottom=0.12)

    y = list(range(len(recs)))[::-1]
    vals   = [v for _, v, _ in recs]
    models = [m for m, _, _ in recs]
    bars = ax.barh(y, vals,
                   color=[COLORS[m] for m in models],
                   edgecolor="white", linewidth=0.8, height=0.72)

    vmax = max(vals)
    for bar, (m, v, c) in zip(bars, recs):
        sub = f"peak @ concurrency={c}  ·  {MODEL_PARAMS[m]['active_b']}B active / {MODEL_PARAMS[m]['total_b']}B total"
        if v >= vmax * 0.35:
            ax.text(vmax * 0.012, bar.get_y() + bar.get_height() / 2,
                    sub, va="center", ha="left", fontsize=8.5, color="white")
            ax.text(v + vmax * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{v:,.0f} tok/s",
                    va="center", ha="left", fontsize=11,
                    fontweight="bold", color=INK)
        else:
            yc = bar.get_y() + bar.get_height() / 2
            ax.text(v + vmax * 0.012, yc + 0.14,
                    f"{v:,.0f} tok/s", va="center", ha="left",
                    fontsize=11, fontweight="bold", color=INK)
            ax.text(v + vmax * 0.012, yc - 0.20,
                    sub, va="center", ha="left", fontsize=8.5, color=MUTED)

    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_DISPLAY[m] for m, _, _ in recs],
                       fontsize=10.5, color=INK, fontweight="bold")
    ax.set_xlim(0, vmax * 1.22)
    ax.set_xlabel("Output throughput (tokens/sec, higher is better)", color=MUTED)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    ax.grid(axis="y", visible=False)

    title_block(
        fig,
        "B300 NVL8 Single-Node Peak Throughput — NVFP4, SGLang, TP=8",
        "Output tokens/second, 1k1k profile (ISL=1024, OSL=1024). Sorted by peak.",
    )

    plt.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
