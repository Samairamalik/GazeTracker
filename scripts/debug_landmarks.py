"""Live window that labels iris, eye-corner and lid landmarks by index, to verify them by eye.

Keys: m = toggle mirrored view, q / Esc = quit. Nothing is written to disk.
"""
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
import yaml

from gazeread.features.landmarks import (
    EYES, LEFT_IRIS_CENTER, LEFT_IRIS_RING, RIGHT_IRIS_CENTER, RIGHT_IRIS_RING,
    LandmarkDetector, eye_openness, is_valid,
)

CONSENT = "Webcam is used live for gaze estimation only; no video or images are saved."
COLORS = {"right": (0, 200, 255), "left": (255, 120, 0)}   # BGR: subject-right orange, left blue
POINTS = {
    "right": (RIGHT_IRIS_CENTER, *RIGHT_IRIS_RING, *EYES["right"]),
    "left": (LEFT_IRIS_CENTER, *LEFT_IRIS_RING, *EYES["left"]),
}


def euler_deg(transform: np.ndarray) -> tuple[float, float, float]:
    """Rough (yaw, pitch, roll) in degrees from the rotation part of the 4x4 matrix."""
    r = transform[:3, :3]
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2])))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return yaw, pitch, roll


def main() -> None:
    print(CONSENT)
    cfg = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    eye_open_min = cfg["landmarks"]["eye_open_min"]
    det = LandmarkDetector(str(ROOT / cfg["landmarks"]["model_path"]), cfg["landmarks"]["num_faces"])
    cap = cv2.VideoCapture(cfg["camera"]["index"], cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        sys.exit("Camera failed to open. On macOS enable camera access for this terminal app in "
                 "System Settings > Privacy & Security > Camera, then restart it.")
    mirrored, fps, t_prev = False, 0.0, time.monotonic()
    while True:
        got, frame = cap.read()
        t = time.monotonic()
        if not got:
            continue
        d = det.detect(frame, t)
        w = frame.shape[1]
        view = cv2.flip(frame, 1) if mirrored else frame.copy()
        lms = d.lms_px if d else None
        if lms is not None:
            for eye, idxs in POINTS.items():
                for i in idxs:
                    x, y = lms[i, 0], lms[i, 1]
                    x = w - 1 - x if mirrored else x
                    cv2.circle(view, (int(x), int(y)), 2, COLORS[eye], -1)
                    cv2.putText(view, str(i), (int(x) + 3, int(y) - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, COLORS[eye], 1)
        valid = is_valid(lms, eye_open_min)
        fps = 0.9 * fps + 0.1 / max(t - t_prev, 1e-6)
        t_prev = t
        lines = [f"{'MIRRORED' if mirrored else 'UNMIRRORED'} view  fps {fps:4.1f}",
                 f"valid={valid}  (eye_open_min={eye_open_min})"]
        if lms is not None:
            lines.append(f"open R(subject)={eye_openness(lms, 'right'):.2f}  "
                         f"L(subject)={eye_openness(lms, 'left'):.2f}")
            yaw, pitch, roll = euler_deg(d.transform)
            tx, ty, tz = d.transform[:3, 3]
            lines.append(f"rough yaw {yaw:+.0f} pitch {pitch:+.0f} roll {roll:+.0f} deg  "
                         f"t=({tx:.0f},{ty:.0f},{tz:.0f}) cm")
        else:
            lines.append("no face")
        lines.append("orange = subject's RIGHT eye, blue = LEFT.  m: mirror  q: quit")
        for k, s in enumerate(lines):
            cv2.putText(view, s, (10, 22 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (0, 255, 0) if valid else (0, 0, 255), 1)
        cv2.imshow("debug_landmarks", view)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("m"):
            mirrored = not mirrored
    cap.release()
    det.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
