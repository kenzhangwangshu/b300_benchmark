#!/usr/bin/env python3
"""Plot B300 measured cost vs public API pricing."""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PLOTS = Path("/sessions/eager-brave-carson/mnt/b300_benchmark/analysis/plots")
PLOTS.mkdir(exist_ok=True, parents=True)

# --- MiniMax M2.7 ---
m27_labels = ["B300 vLLM\n@conc 512", "B300 SGLang\n@conc 512", "MiniMax API /\nOpenRouter"]
m27_input  = [0.014, 0.061, 0.30]
m27_output = [0.894, 0.825, 1.20]
colors = ["#2ca02c", "#1f77b4", "#d62728"]

def bar_chart(labels, values, title, ylabel, fname, ref_lines=None):
    fig, ax = plt.subplots(figsize=(8,5))
    bars = ax.bar(labels, values, color=colors)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, v, f"${v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    if ref_lines:
        for y, label in ref_lines:
            ax.axhline(y, linestyle="--", color="gray", alpha=0.5)
            ax.text(len(labels)-0.5, y, f"  {label}", va="center", fontsize=9, color="gray")
    fig.tight_layout(); fig.savefig(PLOTS/fname, dpi=130); plt.close(fig)
    print("wrote", fname)

bar_chart(m27_labels, m27_input,  "M2.7 — $/M input tokens: measured vs retail",  "$ per million input tokens",  "pricing_compare_m27.png")
bar_chart(m27_labels, m27_output, "M2.7 — $/M output tokens: measured vs retail", "$ per million output tokens", "pricing_compare_m27_out.png")

# --- Kimi K2.5 ---
kimi_labels = ["B300 SGLang\n@conc 128", "Together AI", "Moonshot API"]
kimi_input  = [0.171, 0.50, 0.60]
kimi_output = [3.340, 2.80, 2.50]
colors2 = ["#1f77b4", "#ff7f0e", "#d62728"]

def bar_chart2(labels, values, title, ylabel, fname, highlight_problem=False):
    fig, ax = plt.subplots(figsize=(8,5))
    bars = ax.bar(labels, values, color=colors2)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x()+bar.get_width()/2, v, f"${v:.3f}", ha="center", va="bottom", fontsize=10)
    if highlight_problem:
        bars[0].set_edgecolor("red"); bars[0].set_linewidth(3)
        ax.text(0, values[0]*1.04, "HIGHER than retail", ha="center", color="red", fontsize=9, fontweight="bold")
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(PLOTS/fname, dpi=130); plt.close(fig)
    print("wrote", fname)

bar_chart2(kimi_labels, kimi_input,  "Kimi K2.5 — $/M input tokens: measured vs retail",  "$ per million input tokens",  "pricing_compare_kimi.png")
bar_chart2(kimi_labels, kimi_output, "Kimi K2.5 — $/M output tokens: measured vs retail", "$ per million output tokens", "pricing_compare_kimi_out.png", highlight_problem=True)

# --- Gross margin at cluster scale ---
fig, ax = plt.subplots(figsize=(10,5.5))
scenarios = ["M2.7\n@ OpenRouter $1.50", "M2.7\n@ M2.5-floor $1.26",
             "Kimi K2.5\n@ Moonshot $3.10", "Kimi K2.5\n@ Together $3.30"]
revenue = [1.500, 1.260, 3.100, 3.300]
cost    = [0.886, 0.886, 3.511, 3.511]
tok_per_day_B = [85.3, 85.3, 21.5, 21.5]  # B-tokens/day of output

# daily margin in $ = (revenue - cost) × tok_per_day_B * 1e9 / (2 * 1e6)  [because 1 unit = 1M in + 1M out = 2M tokens total; tok_per_day_B counts output only — so multiply by same]
# simpler: gross = (revenue - cost) per (1M output tokens paired with 1M input tokens); daily = gross × output_Mtok_per_day
# 85.3 B output tok/day = 85,300 M-output-tok/day; each unit pairs with 1 M input
daily_margin = [(r-c) * (t*1000) for r,c,t in zip(revenue, cost, tok_per_day_B)]
# actually (r-c) is $ per (1M in + 1M out) unit; each 1M output token day = 1M in too; so multiplier is output_Mtok_per_day
# 85.3 B tok = 85300 M. OK but this pairs 1M output with 1M input, so 1 unit = 1M of each.
# number of units/day = 85300 (for M2.7)
daily_margin = [(r-c) * (t*1000) for r,c,t in zip(revenue, cost, tok_per_day_B)]

colors3 = ["#2ca02c" if x>0 else "#d62728" for x in daily_margin]
bars = ax.bar(scenarios, [x/1000 for x in daily_margin], color=colors3)
for bar, v in zip(bars, daily_margin):
    ax.text(bar.get_x()+bar.get_width()/2, v/1000, f"${v/1000:+,.1f}K/day",
            ha="center", va="bottom" if v>0 else "top", fontsize=10, fontweight="bold")
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Daily gross margin — 96-node cluster ($K/day)")
ax.set_title("Self-host gross margin vs retail @ 768 B300 GPUs, 100% utilization, 1:1 input:output")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(PLOTS/"gross_margin_projection.png", dpi=130)
plt.close(fig)
print("wrote gross_margin_projection.png")
print("DONE")
