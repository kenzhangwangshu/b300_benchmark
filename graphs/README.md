# graphs/ — SemiAnalysis-style charts for the B300 NVFP4 benchmark

Generates six publication-ready PNGs from `results_595/` (primary) and `results/` (driver 590.48 comparison).

## Quickstart
```bash
cd graphs
pip install -r requirements.txt
python3 build_all.py
# open output/
```

Each chart is an independent script and can be re-run alone:
```bash
python3 chart_01_hero_ranking.py
```

## Files

| File | Purpose |
|---|---|
| `theme.py` | SA-style matplotlib rcParams, color palette, title/credit block |
| `data_loader.py` | Reads sweep JSONs into flat row dicts; `peaks()` and `series()` helpers |
| `chart_01_hero_ranking.py` | Horizontal bar: peak output tok/s @ 1k1k, ranked |
| `chart_02_scaling_curves.py` | 3-panel (1k1k / 1k4k / 4k1k) output tok/s vs concurrency, log-x |
| `chart_03_driver_uplift.py` | Driver 590.48 vs 595 grouped bars, delta % labels |
| `chart_04_peak_matrix.py` | Model × profile heatmap of peak tok/s |
| `chart_05_ttft_curves.py` | Median TTFT vs concurrency (log-log) |
| `chart_06_per_gpu_efficiency.py` | Scatter: per-GPU tok/s vs active parameters |
| `build_all.py` | Run every chart script in sequence |
| `output/` | Generated PNGs (180 dpi, tight bbox) |

## Data expectations
- Driver 595: `../results_595/sglang/<model>/<profile>/json/*.json`
- Driver 590.48: `../results/sglang/<model>/json/*.json` (profile inferred from `random_input_len` / `random_output_len`)
- Metrics consumed: `output_throughput`, `max_concurrency`, `median_ttft_ms`, `median_itl_ms`, plus sequence-length fields.

## Model params
Edit `MODEL_PARAMS` in `theme.py` if the active/total parameter counts for any model are off — they feed Chart 1 annotations and Chart 6 positions.

## Style notes
- Palette is deliberately limited: one color per model, plus semantic green (positive delta) / red (negative delta).
- Fonts fall back to DejaVu Sans if Inter / IBM Plex Sans / Helvetica Neue are not installed.
- Titles are rendered with `fig.text(...)` (not `ax.set_title`) so the layout matches SemiAnalysis conventions (left-aligned block above the axes, credit line at bottom).
