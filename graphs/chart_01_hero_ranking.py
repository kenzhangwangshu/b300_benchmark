"""Chart 1 — Hero: B300 NVFP4 Peak Output Throughput (1k/1k decode), ranked."""
import matplotlib.pyplot as plt
from pathlib import Path

from data_loader import load_595, peaks
from theme import (apply_theme, title_block, COLORS, MODEL_DISPLAY, MODEL_PARAMS,
                   INK, MUTED, fmt_tok)

OUT = Path(__file__).parent / "output" / "chart_01_hero_ranking.png"
PROFILE = "1k1k"


def main():
    apply_theme()
    rows = load_595()
    pk = peaks(rows)

    records = []
    for m, profs in pk.items():
        r = profs.get(PROFILE)
        if r is None:
            continue
        records.append((m, r["output_throughput"], r["concurrency"]))
    records.sort(key=lambda x: x[1], reverse=True)

    labels = [MODEL_DISPLAY[m] for m, _, _ in records]
    vals   = [v for _, v, _ in records]
    concs  = [c for _, _, c in records]
    models = [m for m, _, _ in records]

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    fig.subplots_adjust(left=0.035, right=0.96, top=0.80, bottom=0.12)

    y = list(range(len(labels)))[::-1]
    bars = ax.barh(y, vals,
                   color=[COLORS[m] for m in models],
                   edgecolor="white", linewidth=0.8, height=0.72)

    vmax = max(vals)
    for bar, v, c, m in zip(bars, vals, concs, models):
        sub = f"peak @ concurrency={c}  ·  {MODEL_PARAMS[m]['active_b']}B active / {MODEL_PARAMS[m]['total_b']}B total"
        if v >= vmax * 0.35:
            ax.text(vmax * 0.012, bar.get_y() + bar.get_height() / 2,
                    sub, va="center", ha="left", fontsize=8.5, color="white")
            ax.text(v + vmax * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{v:,.0f} tok/s", va="center", ha="left",
                    fontsize=11, fontweight="bold", color=INK)
        else:
            yc = bar.get_y() + bar.get_height() / 2
            ax.text(v + vmax * 0.012, yc + 0.14,
                    f"{v:,.0f} tok/s", va="center", ha="left",
                    fontsize=11, fontweight="bold", color=INK)
            ax.text(v + vmax * 0.012, yc - 0.20,
                    sub, va="center", ha="left", fontsize=8.5, color=MUTED)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10.5, color=INK, fontweight="bold")
    ax.set_xlim(0, max(vals) * 1.22)
    ax.set_xlabel("Output throughput (tokens/sec, higher is better)", color=MUTED)
    ax.tick_params(axis="y", length=0)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    ax.grid(axis="y", visible=False)

    title_block(
        fig,
        "DeepSeek R1 leads the B300 NVFP4 pack at 12.5k tok/s",
        f"Peak output throughput, {PROFILE} decode, 8×B300 SXM6 AC, TP=8, single node. Sorted by peak.",
    )

    OUT.parent.mkdir(exist_ok=True)
    plt.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
