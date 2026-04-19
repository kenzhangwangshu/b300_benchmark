#!/usr/bin/env python3
"""SGLang/vLLM ratio plot — shows crossover points clearly."""
import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("/sessions/eager-brave-carson/mnt/b300_benchmark/analysis")
df = pd.read_csv(OUT/"all_runs.csv")
m27 = df[(df["model"]=="minimax-m2.7") & (df["profile"]=="1k1k")].copy()

sg = m27[m27["framework"]=="sglang"].set_index("concurrency")
vl = m27[m27["framework"]=="vllm"].set_index("concurrency")
idx = sorted(set(sg.index) & set(vl.index))

fig, ax = plt.subplots(figsize=(10,5.5))
ratios = {
    "Throughput (SGLang / vLLM)": [sg.loc[c,"out_tok_s_per_gpu"] / vl.loc[c,"out_tok_s_per_gpu"] for c in idx],
    "TTFT (SGLang / vLLM) — lower better for SGLang": [sg.loc[c,"mean_ttft_ms"] / vl.loc[c,"mean_ttft_ms"] for c in idx],
    "TPOT (SGLang / vLLM) — lower better for SGLang": [sg.loc[c,"mean_tpot_ms"] / vl.loc[c,"mean_tpot_ms"] for c in idx],
}
styles = {"Throughput (SGLang / vLLM)":"-o",
          "TTFT (SGLang / vLLM) — lower better for SGLang":"-s",
          "TPOT (SGLang / vLLM) — lower better for SGLang":"-^"}
for name, vals in ratios.items():
    ax.plot(idx, vals, styles[name], label=name, linewidth=2, markersize=7)
ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xlabel("Concurrency"); ax.set_ylabel("SGLang ÷ vLLM (log scale, 1.0 = tie)")
ax.set_title("SGLang vs vLLM — metric ratios across concurrency (M2.7 NVFP4, TP=8, 1k1k)\nAbove 1.0: SGLang larger · Below 1.0: vLLM larger")
ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=9, loc="best")
# annotate crossovers
for name, vals in ratios.items():
    for i, (c, v) in enumerate(zip(idx, vals)):
        if i>0 and ((vals[i-1]-1)*(v-1) < 0):  # sign change
            ax.annotate(f"crossover ~conc {c}", (c, 1.0), xytext=(10, -20 if "TTFT" in name else 20),
                        textcoords="offset points", fontsize=9, color="red",
                        arrowprops=dict(arrowstyle="->", color="red", alpha=0.6))

fig.tight_layout()
fig.savefig(OUT/"plots/sglang_vllm_ratio.png", dpi=130)
print("wrote sglang_vllm_ratio.png")

# print the ratios as a table for the report
rep = pd.DataFrame({
    "Concurrency": idx,
    "Throughput ratio": [round(r,2) for r in ratios["Throughput (SGLang / vLLM)"]],
    "TTFT ratio": [round(r,2) for r in ratios["TTFT (SGLang / vLLM) — lower better for SGLang"]],
    "TPOT ratio": [round(r,2) for r in ratios["TPOT (SGLang / vLLM) — lower better for SGLang"]],
})
print(rep.to_string(index=False))
