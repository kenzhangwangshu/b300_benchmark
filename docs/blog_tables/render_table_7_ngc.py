"""Table 7 — NGC Container Compatibility Matrix."""
from pathlib import Path
from table_renderer import render_table, ACCENT_GREEN, ACCENT_RED

OUT = Path(__file__).parent / "table_7_ngc_compat.png"

CHECK = "\u2713 Stable"

COLUMNS = [
    {"label": "Container",     "align": "left",   "width": 1.4},
    {"label": "vLLM Version",  "align": "center", "width": 1.1},
    {"label": "Driver 590",    "align": "center", "width": 1.4},
    {"label": "Driver 595",    "align": "center", "width": 1.4},
    {"label": "Notes",         "align": "left",   "width": 2.4},
]

ROWS = [
    {"cells": ["NGC 26.03",       "0.17.1",
               {"text": "Crashes at conc\u226516", "color": ACCENT_RED},
               {"text": CHECK, "color": ACCENT_GREEN},
               "Has DeepGEMM, needs 595+"]},
    {"cells": ["NGC 26.01",       "0.13.0",
               {"text": CHECK, "color": ACCENT_GREEN},
               {"text": CHECK, "color": ACCENT_GREEN},
               "No DeepGEMM, no B300 MoE config"]},
    {"cells": ["pip install vllm","0.19.0",
               {"text": "CUTLASS crash", "color": ACCENT_RED},
               {"text": "CUTLASS crash", "color": ACCENT_RED},
               "SM103 not in prebuilt binaries"]},
]


def main():
    render_table(
        OUT,
        title="NGC Container Compatibility on B300 SXM6 AC",
        subtitle="Three deployment paths, two driver versions.",
        columns=COLUMNS,
        rows=ROWS,
        footnote="B300 SXM6 AC is SM 103a. Bare-metal pip install fails on every driver — always use Docker.",
        width_in=12.5,
    )


if __name__ == "__main__":
    main()
