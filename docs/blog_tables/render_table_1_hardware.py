"""Table 1 — Hardware Spec."""
from pathlib import Path
from table_renderer import render_table

OUT = Path(__file__).parent / "table_1_hardware.png"

COLUMNS = [
    {"label": "Component",     "align": "left", "width": 1.0},
    {"label": "Specification", "align": "left", "width": 2.4},
]

ROWS = [
    {"cells": ["GPU",                  "8\u00d7 NVIDIA B300 SXM6 AC"]},
    {"cells": ["HBM3e per GPU",        "288 GB"]},
    {"cells": ["Total VRAM",           "2,304 GB"]},
    {"cells": ["Memory Bandwidth",     "8 TB/s per GPU"]},
    {"cells": ["NVLink 5",             "1.8 TB/s per GPU"]},
    {"cells": ["TDP",                  "1,100W per GPU (8,800W node)"]},
    {"cells": ["Cooling",              "Air-cooled"]},
    {"cells": ["Compute Capability",   "SM 103a"]},
    {"cells": ["Driver",               "595.58.03"]},
    {"cells": ["CUDA",                 "13.2"]},
]


def main():
    render_table(
        OUT,
        title="B300 SXM6 AC — Hardware Spec",
        subtitle="Single 8-GPU air-cooled node used for every benchmark in this post.",
        columns=COLUMNS,
        rows=ROWS,
        width_in=8.5,
    )


if __name__ == "__main__":
    main()
