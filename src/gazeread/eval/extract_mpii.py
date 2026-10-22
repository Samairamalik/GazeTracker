"""Extract Method A and B features for every MPIIFaceGaze image, cached per participant.

    python -m gazeread.eval.extract_mpii      # `make extract-mpii`; skips participants already done

Uses the live app's own code: MediaPipe FaceLandmarker (still-image mode) ->
features_from_detection (Method A, same validity gate) and L2CSBackend on the MediaPipe face crop
(Method B). Writes data/processed/mpii/pXX.npz with numbers only (no images or crops).
"""
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from gazeread.data.mpiifacegaze import PARTICIPANTS, load_participant
from gazeread.features.landmarks import LandmarkDetector
from gazeread.gaze.backend_l2cs import L2CSBackend
from gazeread.gaze.backend_landmark import FEATURE_NAMES, features_from_detection

ROOT = Path(__file__).resolve().parents[3]


def extract(pid: str, cfg: dict, app: dict, det: LandmarkDetector, l2cs: L2CSBackend, out: Path) -> dict:
    P = load_participant(ROOT / cfg["dataset_dir"], pid)
    n = len(P.image_paths)
    A = np.full((n, len(FEATURE_NAMES)), np.nan)
    B = np.full((n, 4), np.nan)
    valid = np.zeros(n, bool)
    face = np.zeros(n, bool)
    limits = (app["landmarks"]["eye_open_min"], app["pose"]["max_yaw_deg"], app["pose"]["max_pitch_deg"])
    t0 = time.time()
    for i, path in enumerate(P.image_paths):
        frame = cv2.imread(str(path))
        if frame is None:
            raise FileNotFoundError(path)
        d = det.detect(frame)
        A[i], valid[i] = features_from_detection(d, *limits)
        face[i] = d is not None
        if d is not None:
            B[i] = l2cs.from_detection(frame, d, valid[i], 0.0).vec
        if (i + 1) % 500 == 0:
            print(f"  {pid}: {i + 1}/{n} ({(time.time() - t0) / (i + 1) * 1000:.0f} ms/image)", flush=True)
    np.savez(out, landmark=A, l2cs=B, valid=valid, face=face, target_mm=P.target_mm(),
             target_frac=P.target_frac(), screen_mm=np.array(P.screen_mm), screen_px=np.array(P.screen_px))
    return {"n": n, "face": face.mean(), "valid": valid.mean(), "s": time.time() - t0}


def main() -> None:
    cfg = yaml.safe_load((ROOT / "configs/eval_lopo.yaml").read_text())
    app = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    out_dir = ROOT / cfg["processed_dir"] / "mpii"
    out_dir.mkdir(parents=True, exist_ok=True)
    det = LandmarkDetector(str(ROOT / app["landmarks"]["model_path"]), video=False)
    l2cs = L2CSBackend(str(ROOT / app["l2cs"]["weights"]), app["l2cs"]["crop_scale"], app["l2cs"]["device"])
    print(f"L2CS on {l2cs.device}; writing {out_dir}", flush=True)
    for pid in PARTICIPANTS:
        out = out_dir / f"{pid}.npz"
        if out.exists():
            print(f"{pid}: cached", flush=True)
            continue
        r = extract(pid, cfg, app, det, l2cs, out)
        print(f"{pid}: {r['n']} images, face {100 * r['face']:.1f}%, valid {100 * r['valid']:.1f}%, "
              f"{r['s'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
