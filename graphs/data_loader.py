"""Load benchmark JSONs into flat row dicts. Exposes 595 (primary) + 590 (comparison)."""
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS_595 = REPO / "results_595" / "sglang"
RESULTS_590 = REPO / "results" / "sglang"

MODEL_KEYS = [
    "deepseek-r1",
    "qwen3.5-397b-a17b",
    "minimax-m2.7",
    "glm-5.1",
    "kimi-k2.5",
]

PROFILES = ["1k1k", "1k4k", "4k1k"]

_FIELDS = [
    "output_throughput", "request_throughput", "total_throughput",
    "mean_ttft_ms", "median_ttft_ms", "p99_ttft_ms",
    "mean_itl_ms", "median_itl_ms", "p99_itl_ms",
    "mean_tpot_ms", "median_tpot_ms",
    "mean_e2e_latency_ms", "median_e2e_latency_ms",
    "completed", "duration",
    "total_input_tokens", "total_output_tokens",
    "random_input_len", "random_output_len",
]


def _infer_profile(d):
    ri, ro = d.get("random_input_len"), d.get("random_output_len")
    if ri and ro:
        return f"{ri // 1024}k{ro // 1024}k"
    return "unknown"


def _row(path, model, profile_hint=None):
    with open(path) as f:
        d = json.load(f)
    row = {
        "_path":        str(path),
        "model":        model,
        "profile":      profile_hint or _infer_profile(d),
        "concurrency":  d.get("max_concurrency"),
    }
    for k in _FIELDS:
        row[k] = d.get(k)
    return row


def load_595():
    rows = []
    for m in MODEL_KEYS:
        for p in PROFILES:
            jdir = RESULTS_595 / m / p / "json"
            if not jdir.exists():
                continue
            for jf in sorted(jdir.glob("*.json")):
                rows.append(_row(jf, m, profile_hint=p))
    return rows


def load_590():
    """Driver 590.48 lives under results/sglang/<model>/json/ — profile inferred from JSON."""
    rows = []
    for m in MODEL_KEYS:
        jdir = RESULTS_590 / m / "json"
        if not jdir.exists():
            continue
        for jf in sorted(jdir.glob("*.json")):
            rows.append(_row(jf, m))
    return rows


def peaks(rows):
    """Best output_throughput per (model, profile)."""
    best = defaultdict(lambda: defaultdict(lambda: None))
    for r in rows:
        if r["output_throughput"] is None:
            continue
        cur = best[r["model"]][r["profile"]]
        if cur is None or r["output_throughput"] > cur["output_throughput"]:
            best[r["model"]][r["profile"]] = r
    return best


def series(rows, model, profile, metric="output_throughput"):
    """Sorted [(concurrency, metric)] for a single model × profile sweep."""
    pts = [
        (r["concurrency"], r[metric])
        for r in rows
        if r["model"] == model and r["profile"] == profile
        and r["concurrency"] is not None and r[metric] is not None
    ]
    pts.sort(key=lambda x: x[0])
    return pts


if __name__ == "__main__":
    rows = load_595()
    print(f"595 rows: {len(rows)}")
    pk = peaks(rows)
    for m in MODEL_KEYS:
        for p in PROFILES:
            r = pk[m][p]
            if r is None:
                print(f"  {m} / {p}: (missing)")
            else:
                print(f"  {m:22s} / {p}: {r['output_throughput']:8.1f} tok/s @ c={r['concurrency']}")
    print(f"\n590 rows: {len(load_590())}")
