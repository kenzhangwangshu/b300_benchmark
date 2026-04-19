# docs/blog_charts — Substack blog visualizations

Dark-theme SemiAnalysis-style charts produced from the B300 NVFP4 benchmark data
under `results_595/` (primary) and `results/` (driver 590.48 comparison).

All PNGs render at ~150 DPI with `bbox_inches="tight"`. To produce the 300-DPI
print variant, set `savefig.dpi = 300` in `theme_dark.py` and re-run.

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
| `theme_dark.py` | Dark (#1a1a2e) matplotlib rcParams, model palette (R1 #FF6B6B, Qwen #4ECDC4, M2.7 #45B7D1, GLM #96CEB4, Kimi #FFEAA7), title + footer block |
| `_data.py` | Thin shim that re-exports `graphs/data_loader.py` helpers |
| `chart_01_hero_1k1k.py` | Hero horizontal bar, 1k1k peak tok/s; Kimi greyed as outlier |
| `chart_02_scaling_all.py` | 3-panel scaling curves, log-x concurrency, peak + knee markers |
| `chart_03_driver_delta.py` | 590.48 vs 595 grouped bars, sorted by delta, green/red % labels |
| `chart_04_ep_cliff.py` | Dual-axis EP failure chart: throughput (teal) + TPOT (red) with healthy/failure zones and X marker |
| `chart_05_cost_per_mtok.py` | Self-hosted $/MTok horizontal bars with OpenRouter reference line |
| `chart_06_pareto.py` | Per-model Pareto curves: interactivity (tok/s/user) vs throughput/GPU |
| `chart_07_cluster_economics.py` | 96-node cluster revenue vs cost for TP=1 / TP=8 / TP=8·EP=4 deployments |
| `generate_all.py` | Runs every chart script in order |

## Data contract
- Driver 595: `../../results_595/sglang/<model>/<profile>/json/*.json`
- Driver 590.48: `../../results/sglang/<model>/json/*.json`
- Chart 4 (EP cliff), Chart 5 (cost), Chart 7 (economics) use the
  hard-coded tables in the blog spec — edit the `DATA = [...]` constants
  at the top of those scripts if numbers change.

## Style (dark theme)
- Background `#1a1a2e`, text `#E0E0E0`, muted `#9CA3AF`.
- Positive/uplift: `#4ADE80` green. Negative/regression: `#F87171` red.
- Per-model color keys are shared across charts 2, 6, and (via alt-palette) 1, 3.
- Fonts fall back to DejaVu Sans if Inter / Helvetica Neue are not installed.
