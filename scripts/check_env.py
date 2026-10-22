"""M0 environment check: versions, architecture, MPS, model file, camera. Exit 1 on any failure."""
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import mediapipe as mp
import numpy as np
import torch
import yaml

from gazeread.features.landmarks import LandmarkDetector


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    ok = True
    print(f"python      {platform.python_version()}")
    print(f"arch        {platform.machine()}")
    print(f"mediapipe   {mp.__version__}")
    print(f"torch       {torch.__version__}")
    print(f"opencv      {cv2.__version__}")
    print(f"numpy       {np.__version__}")
    print(f"mps         {torch.backends.mps.is_available()}")
    if platform.python_version_tuple()[:2] != ("3", "11"):
        print("FAIL: expected Python 3.11")
        ok = False

    model = ROOT / cfg["landmarks"]["model_path"]
    if not model.exists():
        print(f"model       MISSING at {model} (see README / Makefile `setup`)")
        ok = False
    else:
        try:
            det = LandmarkDetector(str(model))
            det.detect(np.zeros((480, 640, 3), np.uint8), 0.0)
            det.close()
            print(f"model       loads ({model.name})")
        except Exception as e:  # report any load failure, then keep checking
            print(f"model       FAILED to load: {e}")
            ok = False

    l2cs_weights = ROOT / cfg["l2cs"]["weights"]
    if not l2cs_weights.exists():
        print(f"l2cs        weights not downloaded ({l2cs_weights.name}); only needed for BACKEND=l2cs / COMPARE=1")
    else:
        try:
            import time
            from gazeread.gaze.backend_l2cs import L2CSBackend
            be = L2CSBackend(str(l2cs_weights), cfg["l2cs"]["crop_scale"], cfg["l2cs"]["device"])
            frame = np.zeros((480, 640, 3), np.uint8)
            box_lms = np.array([[200.0, 120.0], [440.0, 400.0]])
            be.gaze_angles(frame, box_lms)                     # warm-up
            t0 = time.perf_counter()
            for _ in range(10):
                be.gaze_angles(frame, box_lms)
            ms = (time.perf_counter() - t0) / 10 * 1000
            print(f"l2cs        loads (safe loader, strict keys) on {be.device}; {ms:.1f} ms per face")
        except RuntimeError as e:
            print(f"l2cs        FAILED: {e}")
            ok = False

    cap = cv2.VideoCapture(cfg["camera"]["index"], cv2.CAP_AVFOUNDATION)
    got_frame, frame = cap.read() if cap.isOpened() else (False, None)
    cap.release()
    if got_frame:
        print(f"camera      opens ({frame.shape[1]}x{frame.shape[0]}); frame not saved")
    else:
        print("camera      FAILED to open or read.")
        print("            On macOS: System Settings > Privacy & Security > Camera, enable the app")
        print("            running Python (Terminal, iTerm, VS Code), then restart that app.")
        ok = False

    print("OK" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
