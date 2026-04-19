# docs/blog_charts — Substack blog visualizations

SemiAnalysis-style light-theme charts produced from the B300 NVFP4 benchmark data
under `results_595/` (primary) and `results/` (driver 590.48 comparison).

## Blog section mapping

| Chart | File | Blog section |
|-------|------|--------------|
| 1 | `hero_throughput_1k1k.png`         | Section 1 — The Setup (hero image) |
| 2 | `scaling_curves_all_profiles.png`  | Section 3 — Scaling Behavior |
| 3 | `driver_590_vs_595.png`            | Section 4 — Driver 595 Uplift |
| 4 | `ep_cliff_deepseek_r1.png`         | Section 6 — Engineering Disclosures |
| 5 | `cost_per_mtok.png`                | Section 5 — Cost Analysis |
| 6 | `pareto_frontier_1k1k.png`         | Section 3 — Scaling Behavior (alternate view) |
| 7 | `cluster_economics_m27.png`        | Section 5 — Cost Analysis |

## Regenerate
```bash
cd docs/blog_charts
python3 generate_all.py
# or any single chart:
python3 chart_01_hero_1k1k.py
```

## Files

| File | Purpose |
|---|---|
| `theme.py` | Light matplotlib rcParams, shared model palette, title + footer block |
| `_data.py` | Thin shim that re-exports `graphs/data_loader.py` helpers |
| `chart_01_hero_1k1k.py` | Ranked horizontal bar, 1k1k peak tok/s with param annotations |
| `chart_02_scaling_all.py` | 3-panel scaling curves, log-x concurrency, peak + knee markers |
| `chart_03_driver_delta.py` | 590.48 vs 595 grouped bars, sorted by delta, green/red % labels |
| `chart_04_ep_cliff.py` | Dual-axis EP failure chart with healthy/failure zones + X marker |
| `chart_05_cost_per_mtok.py` | Self-hosted $/MTok horizontal bars with OpenRouter reference line |
| `chart_06_pareto.py` | Per-model Pareto curves: interactivity vs throughput/GPU |
| `chart_07_cluster_economics.py` | TP=1 / TP=8 / TP=8·EP=4 revenue vs cost grouped bars |
| `generate_all.py` | Runs every chart script in order |

## Data contract
- Driver 595: `../../results_595/sglang/<model>/<profile>/json/*.json`
- Driver 590.48: `../../results/sglang/<model>/json/*.json`
- Chart 4 (EP cliff), Chart 5 (cost), Chart 7 (economics) use the hard-coded
  tables in the blog spec — edit the `DATA = [...]` constants at the top of
  those scripts if numbers change.

## Style
- Palette: one color per model, consistent across charts.
  - DeepSeek R1 `#D4574E`, Qwen 3.5 `#E9A23B`, MiniMax M2.7 `#2A9D8F`,
    GLM-5.1 `#4E6E81`, Kimi K2.5 `#8E6F9E`.
- Semantic accents: positive delta `#2E8B57`, negative delta `#C0392B`,
  comparison/neutral `#1F4E79`, baseline bars `#AAAAAA`.
- Fonts fall back to DejaVu Sans if Inter / IBM Plex Sans / Helvetica Neue
  are not installed.
- Titles render as `fig.text(...)` (left-aligned block above the axes) with a
  credit line at the bottom.
- To produce a 300-DPI print variant, set `savefig.dpi = 300` in `theme.py`
  and re-run `generate_all.py`.
