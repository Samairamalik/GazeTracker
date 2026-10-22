"""Collect the live calibration errors from session logs into a committed, numbers-only table.

    python -m gazeread.eval.measured_sigma            # reads sessions/, writes results/tables/

Rows: every successful camera calibration. Calibrations without an `order_seed` predate the
dot-order shuffle fix (a fixed top-to-bottom order let head drift masquerade as vertical gaze),
so they are listed but marked excluded. The summary gives the sigma values and noise-shape
parameters the simulation uses. Synthetic sessions are never read.
"""
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
TABLES = ROOT / "results" / "tables"


def collect(sessions_dir: Path) -> list[dict]:
    rows = []
    for f in sorted(sessions_dir.glob("*/events.jsonl")):
        ev = [json.loads(line) for line in f.open() if line.strip()]
        meta = [e for e in ev if e["type"] == "meta"]
        if not meta or meta[-1].get("source") != "camera":
            continue
        for e in ev:
            if e["type"] != "calib_result" or not e.get("ok"):
                continue
            dm = [d for d in e["dot_median_err_px"] if d is not None]
            rows.append({
                "session": f.parent.name, "variant": e.get("variant", "A"),
                "n_points": e["n_points"], "features": e["features"],
                "sigma_px": round(e["sigma_px"], 1), "rms_x": round(e["rms_x"], 1),
                "rms_y": round(e["rms_y"], 1),
                "dot_median_rms_px": round(float(np.sqrt(np.mean(np.square(dm)))), 1),
                "included": int("order_seed" in e),
            })
    return rows


def summarize(rows: list[dict]) -> dict:
    inc = [r for r in rows if r["included"]]
    a = [r for r in inc if r["variant"] == "A"]
    b = [r for r in inc if r["variant"] == "B"]
    med = lambda v: float(np.median(v))
    return {
        "n_included": len(inc), "n_excluded": len(rows) - len(inc),
        "rms_y_A_median": med([r["rms_y"] for r in a]),
        "rms_y_A_best": float(min(r["rms_y"] for r in a)),
        "rms_y_B_median": med([r["rms_y"] for r in b]) if b else float("nan"),
        "x_over_y": med([r["rms_x"] / r["rms_y"] for r in inc]),
        "systematic_frac": med([r["dot_median_rms_px"] / r["sigma_px"] for r in inc]),
    }


def main() -> None:
    rows = collect(ROOT / "sessions")
    TABLES.mkdir(parents=True, exist_ok=True)
    with (TABLES / "measured_sigma.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    s = summarize(rows)
    with (TABLES / "measured_sigma_summary.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        w.writerows((k, round(v, 3) if isinstance(v, float) else v) for k, v in s.items())
    print(f"{len(rows)} calibrations ({s['n_included']} included) -> {TABLES}")
    for k, v in s.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")


if __name__ == "__main__":
    main()
