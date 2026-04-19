"""Thin adapter: reuse graphs/data_loader.py from the repo root."""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "graphs"))

from data_loader import load_595, load_590, peaks, series, MODEL_KEYS, PROFILES  # noqa: E402,F401
