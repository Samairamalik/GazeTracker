"""Recompute the heatmap of a logged session offline (no webcam needed).

    python scripts/replay_session.py sessions/<id>            # writes sessions/<id>/replay/
    python scripts/replay_session.py sessions/<id> --check    # also require == live heatmap.json
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml

from gazeread.analysis.heatmap import analyze_session, dumps, export
from gazeread.app.session import read_events


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session", type=Path)
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--check", action="store_true", help="exit 1 unless equal to the live heatmap.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    result = analyze_session(read_events(args.session / "events.jsonl"), cfg["analysis"])
    paths = export(result, args.session / "replay", json_name="replay.json")
    print(f"wrote {len(paths)} files to {args.session / 'replay'}")
    print(f"granularity {result['granularity']}, pages {len(result['pages'])}, "
          f"sigma_px {result['sigma_px']}, rms_y {result['rms_y']}, source {result['source']}")
    if args.check:
        live = (args.session / "heatmap.json").read_text(encoding="utf-8")
        same = live == dumps(result)
        print("replay == live heatmap.json:", same)
        return 0 if same else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
