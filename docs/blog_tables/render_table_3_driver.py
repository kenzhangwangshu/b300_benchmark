"""Table 3 — Driver 590.48 vs 595."""
from pathlib import Path
from table_renderer import render_table, ACCENT_GREEN, ACCENT_RED

OUT = Path(__file__).parent / "table_3_driver_compare.png"

COLUMNS = [
    {"label": "Model",              "align": "left",   "width": 1.6},
    {"label": "Driver 590 (tok/s)", "align": "center", "width": 1.3},
    {"label": "Driver 595 (tok/s)", "align": "center", "width": 1.3},
    {"label": "Delta",              "align": "center", "width": 1.0},
]

ROWS = [
    {"cells": ["DeepSeek R1",   "9,891",
               {"text": "12,518", "weight": "bold"},
               {"text": "+26.6%", "weight": "bold", "color": ACCENT_GREEN}]},
    {"cells": ["Qwen 3.5 397B", "10,652", "11,124",
               {"text": "+4.4%", "color": ACCENT_GREEN}]},
    {"cells": ["GLM-5.1",       "8,913",  "8,953",
               {"text": "+0.4%", "color": ACCENT_GREEN}]},
    {"cells": ["MiniMax M2.7",  "10,284", "9,710",
               {"text": "-5.6%", "weight": "bold", "color": ACCENT_RED}]},
    {"cells": ["Kimi K2.5",     "2,595",  "2,523",
               {"text": "-2.8%", "color": ACCENT_RED}]},
]


def main():
    render_table(
        OUT,
        title="Driver 595 Impact — Not All Models Benefit",
        subtitle="Peak 1k1k throughput (tok/s), same hardware, same SGLang build, only the NVIDIA driver differs.",
        columns=COLUMNS,
        rows=ROWS,
        width_in=10.0,
    )


if __name__ == "__main__":
    main()
