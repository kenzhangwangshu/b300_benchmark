# docs/blog_tables — table-shaped PNG renders for the Substack post

Substack's editor mangled inline-styled HTML tables in our test, so the
final deliverable is **PNG images that look like tables** — same layout,
header colors, accent text, row tints. Drag-and-drop into Substack.

## Files

| File | Blog section |
|---|---|
| `table_1_hardware.png`           | 1 — The Setup |
| `table_2_results_matrix.png`     | 2 — The Numbers |
| `table_3_driver_compare.png`     | 4 — Driver 595 Uplift |
| `table_4_cost_per_mtok.png`      | 5 — Cost Analysis |
| `table_5_cluster_economics.png`  | 5 — Cost Analysis |
| `table_6_ep_failure.png`         | 6 — Engineering Disclosures |
| `table_7_ngc_compat.png`         | 6 — Engineering Disclosures |

## Regenerate
```bash
cd docs/blog_tables
python3 generate_all.py
# or any single table:
python3 render_table_3_driver.py
```

## Source layout

| File | Purpose |
|---|---|
| `table_renderer.py` | Shared helper. Draws a table from `columns` / `rows` dicts using matplotlib rectangles + text — no `ax.table()`, full control over per-cell color, weight, alignment, row-level tints, and a footnote line. |
| `render_table_<N>_<slug>.py` | One per table. Holds the data and any cell-level styling overrides. |
| `generate_all.py` | Runs every script in order. |

## Style contract (mirrors the HTML mock-ups)
- Header row: `#1a1a2e` background, `#e0e0e0` bold text, thicker bottom rule.
- Body cells: white default; rows can carry `bg=#EAF5EA` (positive tint) or `bg=#FBEAEA` (negative tint).
- Accent text: positive `#4CAF50`, negative `#F44336`. Per-cell or per-row.
- Subtle grey row separator (`#dddddd`).
- Title (15 pt bold) + subtitle (10 pt muted) above the table; muted footnote below.

## Data verification (driver 595 + 590 numbers)
Driver 595 peaks (Table 2) and driver 590 peaks (Table 3) were verified
against the JSON files under `../../results_595/sglang/<model>/<profile>/json/`
and `../../results/sglang/<model>/json/` using `peaks()` from
`graphs/data_loader.py`. All 10 numbers match the spec.

## Skipped: Table 8 — Reasoning parser TTFT impact
The blog spec asks for an A/B comparison of mean TTFT with and without
`--reasoning-parser`. Per `CLAUDE.md` the project rule is **no parsers in
benchmark runs** (parser-on TTFT inflates by hundreds of ms because the
reasoning tokens are buffered before the first visible token). Every JSON
in `results_595/` was therefore captured parser-OFF, so we have no paired
parser-ON data to chart. To produce Table 8 / Chart 9 we would need a
fresh A/B run on at least one model.
