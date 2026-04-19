"""Table 2 — Main Results Matrix (peak tok/s per model × profile, 1k1k tok/s/GPU)."""
from pathlib import Path
from table_renderer import (render_table, ACCENT_GREEN, ROW_GREEN_BG_LIGHT,
                             MUTED_ROW)

OUT = Path(__file__).parent / "table_2_results_matrix.png"

COLUMNS = [
    {"label": "Model",        "align": "left",   "width": 1.6},
    {"label": "Total Params", "align": "center", "width": 1.0},
    {"label": "Active",       "align": "center", "width": 0.7},
    {"label": "1k1k (tok/s)", "align": "center", "width": 1.1},
    {"label": "1k4k (tok/s)", "align": "center", "width": 1.1},
    {"label": "4k1k (tok/s)", "align": "center", "width": 1.1},
    {"label": "tok/s/GPU",    "align": "center", "width": 1.0},
]

ROWS = [
    {
        "bg": ROW_GREEN_BG_LIGHT,
        "cells": [
            {"text": "DeepSeek R1",   "weight": "bold"},
            "671B", "37B",
            {"text": "12,518", "weight": "bold", "color": ACCENT_GREEN},
            "8,510", "6,529", "1,565",
        ],
    },
    {
        "cells": [
            {"text": "Qwen 3.5 397B", "weight": "bold"},
            "397B", "17B", "11,124", "8,901", "9,151", "1,391",
        ],
    },
    {
        "cells": [
            {"text": "MiniMax M2.7", "weight": "bold"},
            "230B", "10B", "9,710", "8,579", "6,258", "1,214",
        ],
    },
    {
        "cells": [
            {"text": "GLM-5.1", "weight": "bold"},
            "744B", "40B", "8,953", "7,142", "4,356", "1,119",
        ],
    },
    {
        "text_color": MUTED_ROW,
        "cells": [
            {"text": "Kimi K2.5", "weight": "bold"},
            "1T+", "~40B", "2,523", "2,654", "2,277", "315",
        ],
    },
]


def main():
    render_table(
        OUT,
        title="B300 NVFP4 Throughput — 5 Models, 3 Profiles",
        subtitle="Peak output tok/s per profile, SGLang 0.5.10.post1, TP=8, driver 595. tok/s/GPU from the 1k1k peak.",
        columns=COLUMNS,
        rows=ROWS,
        footnote="Source: results_595/sglang/<model>/<profile>/json/. Highest output_throughput across the concurrency sweep selected per (model, profile).",
        width_in=12.0,
    )


if __name__ == "__main__":
    main()
