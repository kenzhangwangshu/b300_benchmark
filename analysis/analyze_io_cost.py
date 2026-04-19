#!/usr/bin/env python3
"""Split cost between input (prefill) and output (decode) tokens.

Method: Attribute total GPU $-spend on each run between prefill and decode
by the measured per-request time each consumes.

For each run:
  t_prefill_per_req = mean_ttft_ms
  t_decode_per_req  = (output_len - 1) * mean_tpot_ms
  prefill_fraction  = t_prefill / (t_prefill + t_decode)

  run_cost  = $4.10/GPU/hr * 8 GPUs * duration_hr
  cost_in   = run_cost * prefill_fraction
  cost_out  = run_cost * (1 - prefill_fraction)

  $/M-input-tok  = cost_in  / (total_input_tokens  / 1e6)
  $/M-output-tok = cost_out / (total_output_tokens / 1e6)

Caveat: mean_ttft at high concurrency includes scheduler queueing, so the
prefill share is over-attributed at saturation. Trust the low-conc numbers
more for "pure compute" per-input-token cost; trust the high-conc numbers
as "what an operator would actually bill at that operating point."
"""
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("/sessions/eager-brave-carson/mnt/b300_benchmark")
OUT = BASE / "analysis"
PLOTS = OUT / "plots"

GPU_HR = 4.10
NGPU = 8
NODES = 96

df = pd.read_csv(OUT/"all_runs.csv")
# infer output_len from profile
df["output_len"] = df["profile"].str.extract(r"\d+k(\d+)k").astype(int) * 1024
df["input_len"]  = df["profile"].str.extract(r"(\d+)k\d+k").astype(int) * 1024

# per-request attribution times (ms)
df["t_prefill_ms_per_req"] = df["mean_ttft_ms"]
df["t_decode_ms_per_req"]  = (df["output_len"] - 1) * df["mean_tpot_ms"]
df["prefill_fraction"]     = df["t_prefill_ms_per_req"] / (df["t_prefill_ms_per_req"] + df["t_decode_ms_per_req"])

# total $ for the run
df["run_cost_$"] = GPU_HR * NGPU * (df["duration_s"] / 3600.0)
df["cost_in_$"]  = df["run_cost_$"] * df["prefill_fraction"]
df["cost_out_$"] = df["run_cost_$"] - df["cost_in_$"]

df["cost_per_mtok_in"]  = df["cost_in_$"]  / (df["total_input_tokens"]  / 1e6)
df["cost_per_mtok_out_split"] = df["cost_out_$"] / (df["total_output_tokens"] / 1e6)
# ratio of output:input cost per token
df["output_to_input_price_ratio"] = df["cost_per_mtok_out_split"] / df["cost_per_mtok_in"]

# Also compute the equivalent "blended all-to-output" number (for cross-reference with prior report)
df["cost_per_mtok_out_blended"] = df["run_cost_$"] / (df["total_output_tokens"] / 1e6)

view = df[["framework","model","profile","concurrency","mean_ttft_ms","mean_tpot_ms",
           "prefill_fraction","cost_per_mtok_in","cost_per_mtok_out_split",
           "output_to_input_price_ratio","cost_per_mtok_out_blended"]].copy()
view = view.sort_values(["model","framework","concurrency"]).reset_index(drop=True)
view.to_csv(OUT/"cost_io_split.csv", index=False)

# pretty print
with pd.option_context("display.width", 200, "display.max_rows", None, "display.float_format", "{:.4f}".format):
    print(view.to_string())

# --- Plot: $/M input tok vs $/M output tok vs concurrency for 1k1k ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
colors = {"sglang":"#1f77b4","vllm":"#d62728"}
marker = {"minimax-m2.7":"o","kimi-k2.5":"s"}

for (model, fw), g in df[df["profile"]=="1k1k"].groupby(["model","framework"]):
    g = g.sort_values("concurrency")
    c = colors.get(fw, None); m = marker.get(model, "o")
    axes[0].plot(g["concurrency"], g["cost_per_mtok_in"], "-"+m, label=f"{model} / {fw}",
                 color=c, linewidth=2, markersize=7)
    axes[1].plot(g["concurrency"], g["cost_per_mtok_out_split"], "-"+m, label=f"{model} / {fw}",
                 color=c, linewidth=2, markersize=7)

for ax, title, ylabel in [(axes[0], "Cost per million INPUT tokens (prefill share)", "$ / M input tokens"),
                          (axes[1], "Cost per million OUTPUT tokens (decode share)", "$ / M output tokens")]:
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xlabel("Concurrency"); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=9)
fig.suptitle("Cost split: input (prefill) vs output (decode) — 1k1k, TP=8, B300 @ $4.10/GPU/hr", fontsize=12)
fig.tight_layout()
fig.savefig(PLOTS/"cost_input_vs_output.png", dpi=130)
plt.close(fig)
print("wrote cost_input_vs_output.png")

# --- Plot: output:input price ratio vs concurrency ---
fig, ax = plt.subplots(figsize=(9,5))
for (model, fw), g in df[df["profile"]=="1k1k"].groupby(["model","framework"]):
    g = g.sort_values("concurrency")
    ax.plot(g["concurrency"], g["output_to_input_price_ratio"], "-o", label=f"{model} / {fw}", linewidth=2)
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xlabel("Concurrency"); ax.set_ylabel("Output : Input cost-per-token ratio")
ax.set_title("How much more expensive is an output token than an input token? (1k1k, B300)")
ax.axhline(3, linestyle=":", color="gray", alpha=0.6, label="OpenAI-style 3× ratio")
ax.axhline(4, linestyle="--", color="gray", alpha=0.6, label="Anthropic-style 4–5× ratio")
ax.grid(True, alpha=0.3, which="both"); ax.legend()
fig.tight_layout(); fig.savefig(PLOTS/"output_input_cost_ratio.png", dpi=130); plt.close(fig)
print("wrote output_input_cost_ratio.png")

# --- Format table ---
fmt = view.copy()
fmt["prefill_fraction"] = fmt["prefill_fraction"].map("{:.1%}".format)
for c in ["cost_per_mtok_in","cost_per_mtok_out_split","cost_per_mtok_out_blended"]:
    fmt[c] = fmt[c].map("${:.3f}".format)
fmt["output_to_input_price_ratio"] = fmt["output_to_input_price_ratio"].map("{:.1f}×".format)
fmt["mean_ttft_ms"] = fmt["mean_ttft_ms"].map("{:.0f}".format)
fmt["mean_tpot_ms"] = fmt["mean_tpot_ms"].map("{:.2f}".format)
fmt.columns = ["Framework","Model","Profile","Conc","TTFT ms","TPOT ms",
               "Prefill %","$/M input","$/M output","Out:In ratio","$/M out (blended)"]

with open(OUT/"cost_io_split_tables.md","w") as f:
    f.write("# Input vs Output Token Cost — Split by Prefill/Decode Time Attribution\n\n")
    f.write("Attribution model: per-request prefill time ≈ mean TTFT; decode time ≈ (output_len-1) × mean TPOT. ")
    f.write("Run cost is split between input and output by the ratio of these times.\n\n")
    for (model, fw), g in fmt.groupby(["Model","Framework"]):
        f.write(f"## {model} / {fw}\n\n")
        f.write(g.drop(columns=["Model","Framework"]).to_markdown(index=False) + "\n\n")

print("wrote cost_io_split_tables.md")
print("DONE")
