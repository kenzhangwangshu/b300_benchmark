"""Table 4 — Cost per Million Output Tokens."""
from pathlib import Path
from table_renderer import render_table, ACCENT_GREEN, ACCENT_RED, MUTED_ROW

OUT = Path(__file__).parent / "table_4_cost_per_mtok.png"

COLUMNS = [
    {"label": "Model",              "align": "left",   "width": 1.4},
    {"label": "Peak tok/s",         "align": "center", "width": 0.9},
    {"label": "tok/s per GPU",      "align": "center", "width": 0.9},
    {"label": "$/MTok @ $4.10/hr",  "align": "center", "width": 1.1},
    {"label": "$/MTok @ $6.50/hr",  "align": "center", "width": 1.1},
    {"label": "vs OpenRouter",      "align": "center", "width": 1.2},
]

ROWS = [
    {"cells": ["DeepSeek R1",   "12,518", "1,565",
               {"text": "$0.73", "weight": "bold"}, "$1.15",
               {"text": "67% cheaper", "color": ACCENT_GREEN}]},
    {"cells": ["Qwen 3.5 397B", "11,124", "1,391", "$0.82", "$1.30", "—"]},
    {"cells": ["MiniMax M2.7",   "9,710", "1,214", "$0.94", "$1.49", "—"]},
    {"cells": ["GLM-5.1",        "8,953", "1,119", "$1.02", "$1.61", "—"]},
    {"text_color": MUTED_ROW,
     "cells":     ["Kimi K2.5",  "2,523", "315",   "$3.61", "$5.73",
                   {"text": "more expensive", "color": ACCENT_RED}]},
]


def main():
    render_table(
        OUT,
        title="Self-Hosted B300 Cost vs Public API",
        subtitle="$ per million output tokens at peak throughput, two GPU rates.",
        columns=COLUMNS,
        rows=ROWS,
        footnote="Assumes 100% utilization. Formula: (8 × hourly rate) ÷ (peak tok/s × 3600) × 1,000,000. OpenRouter DeepSeek R1 output: $2.19/MTok as of April 2026.",
        width_in=12.5,
    )


if __name__ == "__main__":
    main()
