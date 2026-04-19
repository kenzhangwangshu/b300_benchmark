"""Table 6 — EP Failure Data (DeepSeek R1, EP enabled, concurrency sweep)."""
from pathlib import Path
from table_renderer import (render_table, ACCENT_GREEN, ACCENT_RED,
                             ROW_RED_BG_LIGHT)

OUT = Path(__file__).parent / "table_6_ep_failure.png"

COLUMNS = [
    {"label": "Concurrency",         "align": "center", "width": 1.0},
    {"label": "Throughput (tok/s)",  "align": "center", "width": 1.3},
    {"label": "TTFT (ms)",           "align": "center", "width": 1.0},
    {"label": "TPOT (ms)",           "align": "center", "width": 1.0},
    {"label": "Failed",              "align": "center", "width": 1.0},
]

ROWS = [
    {"cells": ["1",  "134",                  "359", "7.1",  {"text": "0", "color": ACCENT_GREEN}]},
    {"cells": ["2",  "258",                  "67",  "7.7",  {"text": "0", "color": ACCENT_GREEN}]},
    {"cells": ["4",  "469",                  "75",  "8.5",  {"text": "0", "color": ACCENT_GREEN}]},
    {"cells": ["8",  {"text": "815", "weight": "bold"},
                     "421", "9.4",  {"text": "0", "color": ACCENT_GREEN}]},
    {"bg": ROW_RED_BG_LIGHT,
     "cells": [{"text": "16", "weight": "bold"},
               {"text": "146 \u26A0", "weight": "bold", "color": ACCENT_RED},
               {"text": "843",   "color": ACCENT_RED},
               {"text": "1,425", "weight": "bold", "color": ACCENT_RED},
               {"text": "96/160","weight": "bold", "color": ACCENT_RED}]},
]


def main():
    render_table(
        OUT,
        title="Expert Parallelism Cliff — DeepSeek R1 NVFP4",
        subtitle="EP works up to concurrency 8. At concurrency 16 the deployment collapses.",
        columns=COLUMNS,
        rows=ROWS,
        footnote="SGLang TP=8, EP enabled. 60% of requests returned Internal Server Error at conc=16; TPOT exploded ~150×.",
        width_in=11.0,
    )


if __name__ == "__main__":
    main()
