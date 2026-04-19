"""Regenerate every blog chart in this directory."""
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

MODULES = [
    "chart_01_hero_1k1k",
    "chart_02_scaling_all",
    "chart_03_driver_delta",
    "chart_04_ep_cliff",
    "chart_05_cost_per_mtok",
    "chart_06_pareto",
    "chart_07_cluster_economics",
]


def main():
    for name in MODULES:
        print(f"=== {name} ===")
        importlib.import_module(name).main()
    print("\nDone. All PNGs saved next to this script.")


if __name__ == "__main__":
    main()
