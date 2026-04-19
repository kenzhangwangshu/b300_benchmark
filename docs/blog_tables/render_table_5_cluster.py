"""Table 5 — Cluster Economics (MiniMax M2.7, 96-node cluster)."""
from pathlib import Path
from table_renderer import (render_table, ACCENT_GREEN, ROW_GREEN_BG_LIGHT)

OUT = Path(__file__).parent / "table_5_cluster_economics.png"

COLUMNS = [
    {"label": "Config",              "align": "left",   "width": 1.7},
    {"label": "GPUs/Instance",       "align": "center", "width": 0.9},
    {"label": "Replicas",            "align": "center", "width": 0.8},
    {"label": "Cluster tok/s",       "align": "center", "width": 1.0},
    {"label": "Revenue/hr",          "align": "center", "width": 1.0},
    {"label": "Cost/hr",             "align": "center", "width": 0.9},
    {"label": "Gross Margin",        "align": "center", "width": 1.1},
]

ROWS = [
    {"bg": ROW_GREEN_BG_LIGHT,
     "cells": [{"text": "TP=1 (horizontal)", "weight": "bold"},
               "1",
               {"text": "768",   "weight": "bold"},
               {"text": "7.2M",  "weight": "bold"},
               "$21.7K", "$3.1K",
               {"text": "85.5%", "weight": "bold", "color": ACCENT_GREEN}]},
    {"cells": ["TP=8 (per-node)", "8", "96", "~1.5M", "$4.6K", "$3.1K", "31.4%"]},
    {"cells": ["TP=8 EP=4 (multi-node)", "32", "24", "~1.5M",
               "$4.6K", "$3.1K", "31.4%"]},
]


def main():
    render_table(
        OUT,
        title="MiniMax M2.7 Cluster Economics — Horizontal Scaling Wins",
        subtitle="96-node B300 cluster (768 GPUs). Same hardware, three deployment shapes.",
        columns=COLUMNS,
        rows=ROWS,
        footnote="Revenue at OpenRouter pricing ($0.30/MTok input, $1.20/MTok output). GPU cost at $4.10/GPU-hr. TP=1 is possible because NVFP4 weights (~115 GB) fit in one B300 GPU (288 GB).",
        width_in=12.5,
    )


if __name__ == "__main__":
    main()
