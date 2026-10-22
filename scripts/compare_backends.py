"""Paired comparison of calibration variants A, B and B+pose from compare-mode session logs.

    python scripts/compare_backends.py sessions/<id1> [sessions/<id2> ...] [--include-synthetic]

Each calibration in compare mode fits and validates every variant on the same dots and frames,
so differences within one calibration are paired. Prints every calibration separately, then the
spread over calibrations (min / median / max) and the per-calibration paired differences.
Writes sessions/<id>/compare.json. All numbers come from the logs.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gazeread.app.session import read_events

VARIANTS = ("A", "B", "B+pose")
COLS = ("sigma_px", "rms_x", "rms_y", "loo_rms_px")


def session_rows(path: Path) -> tuple[str, list[dict], dict]:
    """(source, calibrations, per-backend frame stats) for one session directory."""
    ev = read_events(path / "events.jsonl")
    meta = [e for e in ev if e["type"] == "meta"]
    source = meta[-1].get("source", "?") if meta else "?"
    calibs = defaultdict(dict)
    for e in ev:
        if e["type"] == "calib_result" and "variant" in e:
            calibs[e["order_seed"]][e["variant"]] = e
    stats = {}
    for b in ("landmark", "l2cs"):
        f = [e for e in ev if e["type"] == "features" and e.get("backend", "landmark") == b]
        if len(f) > 1:
            t = np.array([e["t"] for e in f])
            stats[b] = {"frames": len(f), "fps": round((len(f) - 1) / (t[-1] - t[0]), 1),
                        "valid_pct": round(100 * np.mean([e["valid"] for e in f]), 1)}
    rows = []
    for seed, by_variant in calibs.items():
        row = {"order_seed": seed}
        for v in VARIANTS:
            r = by_variant.get(v)
            if r is None:
                continue
            row[v] = ({k: round(r[k], 1) for k in COLS} | {"degree": r["degree"]}) if r.get("ok") \
                else {"failed": r.get("reason", "?")}
        rows.append(row)
    return source, rows, stats


def fmt(cell: dict | None) -> str:
    if cell is None:
        return "-"
    if "failed" in cell:
        return f"FAILED ({cell['failed']})"
    return (f"sigma {cell['sigma_px']:6.1f}  x {cell['rms_x']:6.1f}  y {cell['rms_y']:6.1f}  "
            f"LOO {cell['loo_rms_px']:6.1f}  deg {cell['degree']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sessions", nargs="+", type=Path)
    ap.add_argument("--include-synthetic", action="store_true",
                    help="include synthetic sessions in the spread (they are never accuracy results)")
    args = ap.parse_args()

    all_rows = []
    for s in args.sessions:
        source, rows, stats = session_rows(s)
        print(f"\n== {s}  (source: {source}{', NOT a measurement' if source == 'synthetic' else ''})")
        for b, st in stats.items():
            print(f"   {b:8s} frames {st['frames']}  fps {st['fps']}  valid {st['valid_pct']}%")
        if not rows:
            print("   no compare-mode calibrations in this session")
        for i, row in enumerate(rows, 1):
            print(f"   calibration {i} (seed {row['order_seed']})")
            for v in VARIANTS:
                print(f"     {v:7s} {fmt(row.get(v))}")
        (s / "compare.json").write_text(json.dumps({"source": source, "stats": stats, "calibrations": rows},
                                                   indent=1), encoding="utf-8")
        if source != "synthetic" or args.include_synthetic:
            all_rows += [r | {"session": s.name} for r in rows]

    ok = lambda r, v: v in r and "sigma_px" in r[v]
    print(f"\n== spread over {len(all_rows)} calibration(s)")
    for v in VARIANTS:
        vals = [r[v]["sigma_px"] for r in all_rows if ok(r, v)]
        if vals:
            print(f"   {v:7s} sigma_px  n={len(vals)}  min {min(vals):.1f}  median {np.median(vals):.1f}  max {max(vals):.1f}")
    for v in ("B", "B+pose"):
        diffs = [(r["session"], r[v]["sigma_px"] - r["A"]["sigma_px"]) for r in all_rows if ok(r, v) and ok(r, "A")]
        if diffs:
            d = [x for _, x in diffs]
            print(f"   {v} - A (paired, px; negative = {v} better): "
                  + ", ".join(f"{x:+.1f}" for x in d)
                  + f"  | {sum(x < 0 for x in d)}/{len(d)} better, median {np.median(d):+.1f}")
    if len(all_rows) < 3:
        print("   (fewer than 3 calibrations: treat any difference as anecdotal)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
