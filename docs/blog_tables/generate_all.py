"""Render every blog table PNG. Run from inside docs/blog_tables/."""
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

MODULES = [
    "render_table_1_hardware",
    "render_table_2_results",
    "render_table_3_driver",
    "render_table_4_cost",
    "render_table_5_cluster",
    "render_table_6_ep",
    "render_table_7_ngc",
]


def main():
    for name in MODULES:
        print(f"=== {name} ===")
        importlib.import_module(name).main()
    print("\nDone. All PNGs saved next to this script.")


if __name__ == "__main__":
    main()
