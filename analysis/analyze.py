#!/usr/bin/env python3
"""B300 benchmark analysis: parse JSONs, build tables, plot Pareto curves, compute economics."""
import json, os, glob, re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE = Path("/sessions/eager-brave-carson/mnt/b300_benchmark")
OUT = BASE / "analysis"
PLOTS = OUT / "plots"
PLOTS.mkdir(parents=True, exist_ok=True)

GPU_HR_COST = 4.10  # $/GPU/hr
NUM_GPUS = 8
NODES = 96

rows = []
for path in glob.glob(str(BASE / "results/*/*/json/*.json")):
    p = Path(path)
    parts = p.parts
    framework = parts[-4]  # sglang / vllm
    model = parts[-3]
    fn = p.stem
    # skip the stray fp8 tp4 files (not part of NVFP4 tp8 sweep)
    if "fp8_tp4" in fn:
        continue
    m = re.search(r"conc(\d+)_(\d+k\d+k)", fn)
    if not m:
        continue
    conc = int(m.group(1))
    profile = m.group(2)
    with open(path) as f:
        d = json.load(f)
    # unified field names
    total_out = d.get("total_output_tokens") or d.get("completed", 0) * d.get("random_output_len", 0)
    total_in = d.get("total_input_tokens", 0)
    duration = d.get("duration", 0)
    # output throughput (generation tokens/sec)
    out_tp = d.get("output_throughput", 0)
    total_tp = d.get("total_throughput") or d.get("total_token_throughput", 0)
    mean_tpot = d.get("mean_tpot_ms", float("nan"))
    mean_ttft = d.get("mean_ttft_ms", float("nan"))
    p99_ttft = d.get("p99_ttft_ms", float("nan"))
    mean_e2e = d.get("mean_e2e_latency_ms", float("nan"))
    mean_itl = d.get("mean_itl_ms", float("nan"))
    completed = d.get("completed", 0)
    failed = d.get("failed", 0)
    rows.append({
        "framework": framework,
        "model": model,
        "profile": profile,
        "concurrency": conc,
        "duration_s": duration,
        "completed": completed,
        "failed": failed,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "output_tok_s": out_tp,
        "total_tok_s": total_tp,
        "out_tok_s_per_gpu": out_tp / NUM_GPUS,
        "total_tok_s_per_gpu": total_tp / NUM_GPUS,
        "mean_tpot_ms": mean_tpot,
        "mean_ttft_ms": mean_ttft,
        "p99_ttft_ms": p99_ttft,
        "mean_e2e_ms": mean_e2e,
        "mean_itl_ms": mean_itl,
        "interactivity": 1000.0 / mean_tpot if mean_tpot and mean_tpot > 0 else float("nan"),
        # cost: $/M output tokens = ($/hr * hrs) / (Mtokens/hr). hrs to produce 1M = 1e6/(out_tok_s*3600)
        # So $/Mtok = GPU_HR_COST * NUM_GPUS / (out_tok_s * 3600 / 1e6) = GPU_HR_COST*NUM_GPUS*1e6/(out_tok_s*3600)
        "cost_per_mtok_out": (GPU_HR_COST * NUM_GPUS * 1e6) / (out_tp * 3600) if out_tp else float("nan"),
        "cost_per_mtok_total": (GPU_HR_COST * NUM_GPUS * 1e6) / (total_tp * 3600) if total_tp else float("nan"),
    })

df = pd.DataFrame(rows).sort_values(["model","framework","profile","concurrency"]).reset_index(drop=True)
df.to_csv(OUT/"all_runs.csv", index=False)
print(df.to_string())

# --- Pareto plots: throughput/GPU (y) vs interactivity = 1000/tpot (x) ---
def pareto_plot(sub, title, fname, label_col="framework"):
    fig, ax = plt.subplots(figsize=(9,6))
    colors = {"sglang":"#1f77b4", "vllm":"#d62728", "minimax-m2.7":"#1f77b4", "kimi-k2.5":"#2ca02c"}
    for label, g in sub.groupby(label_col):
        g = g.sort_values("concurrency")
        c = colors.get(label, None)
        ax.plot(g["interactivity"], g["out_tok_s_per_gpu"], "-o", label=label, color=c, linewidth=2, markersize=7)
        for _, r in g.iterrows():
            ax.annotate(f"c{int(r['concurrency'])}", (r["interactivity"], r["out_tok_s_per_gpu"]),
                        textcoords="offset points", xytext=(6,4), fontsize=8, alpha=0.7)
    ax.set_xlabel("Interactivity (1000 / mean TPOT ms) — tokens/sec per user")
    ax.set_ylabel("Throughput (output tokens/sec per GPU)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS/fname, dpi=130)
    plt.close(fig)
    print("wrote", fname)

# Framework comparison on M2.7 1k1k
m27 = df[(df["model"]=="minimax-m2.7") & (df["profile"]=="1k1k")]
pareto_plot(m27, "MiniMax M2.7 NVFP4 — SGLang vs vLLM (1k1k, TP=8, B300)", "pareto_m27_framework.png", "framework")

# All SGLang models on 1k1k
sg = df[(df["framework"]=="sglang") & (df["profile"]=="1k1k")]
pareto_plot(sg, "SGLang models — Pareto (1k1k, TP=8, B300)", "pareto_sglang_models.png", "model")

# --- Throughput vs concurrency (log x) ---
def tput_vs_conc(sub, title, fname, label_col="framework"):
    fig, ax = plt.subplots(figsize=(9,5))
    for label, g in sub.groupby(label_col):
        g = g.sort_values("concurrency")
        ax.plot(g["concurrency"], g["out_tok_s_per_gpu"], "-o", label=label, linewidth=2)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Concurrency")
    ax.set_ylabel("Output tokens/sec per GPU")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS/fname, dpi=130)
    plt.close(fig)
    print("wrote", fname)

tput_vs_conc(m27, "M2.7 throughput/GPU vs concurrency (1k1k)", "tput_vs_conc_m27.png", "framework")
tput_vs_conc(sg, "SGLang throughput/GPU vs concurrency (1k1k)", "tput_vs_conc_sglang.png", "model")

# --- TTFT vs concurrency (shows SGLang prefill bottleneck) ---
fig, ax = plt.subplots(figsize=(9,5))
for label, g in m27.groupby("framework"):
    g = g.sort_values("concurrency")
    ax.plot(g["concurrency"], g["mean_ttft_ms"], "-o", label=f"{label} (mean)", linewidth=2)
    ax.plot(g["concurrency"], g["p99_ttft_ms"], "--s", label=f"{label} (p99)", alpha=0.5)
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xlabel("Concurrency"); ax.set_ylabel("TTFT (ms)")
ax.set_title("M2.7 TTFT vs concurrency — SGLang vs vLLM (1k1k)")
ax.grid(True, alpha=0.3, which="both"); ax.legend()
fig.tight_layout(); fig.savefig(PLOTS/"ttft_vs_conc_m27.png", dpi=130); plt.close(fig)
print("wrote ttft_vs_conc_m27.png")

# --- Cost per M output tokens vs concurrency ---
fig, ax = plt.subplots(figsize=(9,5))
for (model, fw), g in df[df["profile"]=="1k1k"].groupby(["model","framework"]):
    g = g.sort_values("concurrency")
    ax.plot(g["concurrency"], g["cost_per_mtok_out"], "-o", label=f"{model} / {fw}", linewidth=2)
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xlabel("Concurrency"); ax.set_ylabel("$ per million output tokens")
ax.set_title(f"Cost per M output tokens (1k1k, ${GPU_HR_COST:.2f}/GPU/hr × 8 GPUs)")
ax.grid(True, alpha=0.3, which="both"); ax.legend()
fig.tight_layout(); fig.savefig(PLOTS/"cost_vs_conc.png", dpi=130); plt.close(fig)
print("wrote cost_vs_conc.png")

# --- Save compact summary tables (markdown) ---
def fmt_table(sub, cols):
    s = sub[cols].copy()
    fmts = {
        "concurrency":"{:d}","out_tok_s_per_gpu":"{:.0f}","total_tok_s_per_gpu":"{:.0f}",
        "mean_tpot_ms":"{:.2f}","mean_ttft_ms":"{:.0f}","p99_ttft_ms":"{:.0f}",
        "interactivity":"{:.1f}","cost_per_mtok_out":"${:.2f}","cost_per_mtok_total":"${:.2f}",
        "mean_itl_ms":"{:.2f}","mean_e2e_ms":"{:.0f}",
    }
    for c,f in fmts.items():
        if c in s.columns:
            s[c] = s[c].map(lambda x: f.format(x) if pd.notna(x) else "—")
    return s.to_markdown(index=False)

tables = {}
for (model, fw), g in df.groupby(["model","framework"]):
    for profile, g2 in g.groupby("profile"):
        key = f"{model}__{fw}__{profile}"
        tables[key] = fmt_table(g2.sort_values("concurrency"),
            ["concurrency","out_tok_s_per_gpu","mean_tpot_ms","mean_ttft_ms","p99_ttft_ms","interactivity","cost_per_mtok_out"])

# Cluster economics at 96 nodes = 768 GPUs
# use best sustained tput/GPU per (model, framework, profile)
best = df.sort_values("out_tok_s_per_gpu", ascending=False).groupby(["model","framework","profile"]).head(1).sort_values(["model","framework"])
best["cluster_tok_s"] = best["out_tok_s_per_gpu"] * NUM_GPUS * NODES
best["cluster_tok_day_billions"] = best["cluster_tok_s"] * 86400 / 1e9
best["cluster_mtok_hr"] = best["cluster_tok_s"] * 3600 / 1e6
best["cluster_cost_hr"] = GPU_HR_COST * NUM_GPUS * NODES
best["cluster_cost_day"] = best["cluster_cost_hr"] * 24
economics = best[["model","framework","profile","concurrency","out_tok_s_per_gpu",
                  "cluster_tok_s","cluster_mtok_hr","cluster_tok_day_billions",
                  "cost_per_mtok_out","cluster_cost_hr","cluster_cost_day"]].copy()
economics.to_csv(OUT/"cluster_economics.csv", index=False)
print("\nCluster economics:")
print(economics.to_string())

# Save formatted economics table
econ_fmt = economics.copy()
econ_fmt["out_tok_s_per_gpu"] = econ_fmt["out_tok_s_per_gpu"].map("{:.0f}".format)
econ_fmt["cluster_tok_s"]     = econ_fmt["cluster_tok_s"].map(lambda x: f"{x:,.0f}")
econ_fmt["cluster_mtok_hr"]   = econ_fmt["cluster_mtok_hr"].map("{:.1f}".format)
econ_fmt["cluster_tok_day_billions"] = econ_fmt["cluster_tok_day_billions"].map("{:.1f}".format)
econ_fmt["cost_per_mtok_out"] = econ_fmt["cost_per_mtok_out"].map("${:.3f}".format)
econ_fmt["cluster_cost_hr"]   = econ_fmt["cluster_cost_hr"].map("${:,.0f}".format)
econ_fmt["cluster_cost_day"]  = econ_fmt["cluster_cost_day"].map("${:,.0f}".format)
econ_fmt.columns = ["Model","Framework","Profile","Peak Conc","Peak tok/s/GPU",
                    "Cluster tok/s (768 GPU)","Cluster M-tok/hr","Cluster B-tok/day",
                    "$/M-tok","Cluster $/hr","Cluster $/day"]
econ_md = econ_fmt.to_markdown(index=False)

with open(OUT/"tables.md","w") as f:
    f.write("# Benchmark Tables\n\n")
    for k, t in tables.items():
        f.write(f"## {k}\n\n{t}\n\n")
    f.write("## Cluster Economics (96 nodes = 768 B300 GPUs)\n\n")
    f.write(econ_md + "\n")
print("wrote tables.md")
print("DONE")
