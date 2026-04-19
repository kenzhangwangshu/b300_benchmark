"""Regenerate every chart. Run from the graphs/ directory:  python3 build_all.py"""
import importlib
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

MODULES = [
    "chart_01_hero_ranking",
    "chart_02_scaling_curves",
    "chart_03_driver_uplift",
    "chart_04_peak_matrix",
    "chart_05_ttft_curves",
    "chart_06_per_gpu_efficiency",
]


def main():
    for name in MODULES:
        print(f"=== {name} ===")
        mod = importlib.import_module(name)
        mod.main()
    print("\nDone. See graphs/output/")


if __name__ == "__main__":
    main()
