"""Chart 1 — Hero bar: peak 1k1k throughput per model."""
from pathlib import Path
import matplotlib.pyplot as plt

from _data import load_595, peaks
from theme_dark import (apply_dark, title_block, MODEL_DISPLAY,
                         TEXT, MUTED, NEUTRAL)

HERO_BLUE = "#4ECDC4"
OUT = Path(__file__).parent / "hero_throughput_1k1k.png"
PROFILE = "1k1k"


def main():
    apply_dark()
    pk = peaks(load_595())
    recs = []
    for m, profs in pk.items():
        r = profs.get(PROFILE)
        if r is None:
            continue
        recs.append((m, r["output_throughput"], r["concurrency"]))
    recs.sort(key=lambda x: x[1], reverse=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.subplots_adjust(left=0.22, right=0.93, top=0.76, bottom=0.16)

    y = list(range(len(recs)))[::-1]
    vals = [v for _, v, _ in recs]
    colors = [NEUTRAL if m == "kimi-k2.5" else HERO_BLUE for m, _, _ in recs]
    bars = ax.barh(y, vals, color=colors, edgecolor="none", height=0.68)

    vmax = max(vals)
    for bar, (m, v, c) in zip(bars, recs):
        ax.text(v + vmax * 0.012, bar.get_y() + bar.get_height() / 2,
                f"{v:,.0f}",
                va="center", ha="left", fontsize=12, fontweight="bold",
                color=TEXT)

    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_DISPLAY[m] for m, _, _ in recs],
                       fontsize=10.5, color=TEXT, fontweight="bold")
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, vmax * 1.17)
    ax.set_xlabel("Output tok/s", color=MUTED)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)

    title_block(
        fig,
        "B300 NVL8 Single-Node Peak Throughput — NVFP4, SGLang, TP=8",
        "Output tokens/second, 1k1k profile (ISL=1024, OSL=1024)",
    )

    plt.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
