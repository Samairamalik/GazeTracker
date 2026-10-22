"""Thin wrapper around MediaPipe FaceLandmarker (Tasks API, mediapipe 1.1.1).

Verified against the installed package and the official Python guide:
- `vision.FaceLandmarkerOptions(..., running_mode=VIDEO, output_facial_transformation_matrixes=True)`
- `landmarker.detect_for_video(mp.Image, timestamp_ms)`; timestamps are int ms and must increase.
- 478 landmarks per face, normalized x, y in [0, 1] of image width/height, z on roughly the x scale.
- `facial_transformation_matrixes[i]` is a 4x4 matrix mapping the canonical face model
  to the detected face; translation (cm) is the last column.

"Right"/"left" eye means the SUBJECT's right/left (subject's right eye appears on the
image's left in an unmirrored frame).
"""
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

# Landmark indices. Verified 2026-10-22 on matplotlib's public-domain grace_hopper.jpg:
# geometric checks (tests/test_landmark_indices.py) plus a visual check of a labeled crop.
# Live webcam (2026-10-22, ~30 fps, numbers only): open eyes ~0.34 openness, blinks ~0.02-0.04,
# 11 blinks gave valid=False runs of 1-11 frames. Right wink: R ~0.06-0.11, L ~0.20-0.26;
# left wink: L ~0.07-0.09, R ~0.22-0.25 (both -> valid=False; the open eye drops too, so margin
# above eye_open_min=0.15 is modest). Covered camera -> no face -> valid=False.
RIGHT_IRIS_CENTER, RIGHT_IRIS_RING = 468, (469, 470, 471, 472)
LEFT_IRIS_CENTER, LEFT_IRIS_RING = 473, (474, 475, 476, 477)
EYES = {
    # eye: (outer corner, inner corner, upper lid, lower lid)
    "right": (33, 133, 159, 145),
    "left": (263, 362, 386, 374),
}
NUM_LANDMARKS = 478


@dataclass
class Detection:
    lms_px: np.ndarray       # (478, 3) landmarks; x, y in image px, z scaled by image width
    transform: np.ndarray    # (4, 4) canonical-face -> detected-face; translation in cm


class LandmarkDetector:
    """FaceLandmarker in VIDEO mode. Call `detect` once per frame with increasing times."""

    def __init__(self, model_path: str, num_faces: int = 1):
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=num_faces,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._last_ms = -1

    def detect(self, frame_bgr: np.ndarray, t_s: float) -> Detection | None:
        """Run on one BGR frame captured at monotonic time `t_s` (seconds). None if no face."""
        t_ms = max(int(t_s * 1000), self._last_ms + 1)   # MediaPipe needs strictly increasing ms
        self._last_ms = t_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, t_ms)
        if not result.face_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        lms = np.array([(p.x * w, p.y * h, p.z * w) for p in result.face_landmarks[0]])
        return Detection(lms_px=lms, transform=np.array(result.facial_transformation_matrixes[0]))

    def close(self) -> None:
        self._landmarker.close()


def eye_openness(lms_px: np.ndarray, eye: str) -> float:
    """Lid distance / eye width (unitless), from landmarks in image px. `eye` is "right" or "left"."""
    outer, inner, upper, lower = EYES[eye]
    width = np.linalg.norm(lms_px[outer, :2] - lms_px[inner, :2])
    lid = np.linalg.norm(lms_px[upper, :2] - lms_px[lower, :2])
    return float(lid / width) if width > 0 else 0.0


def is_valid(lms_px: np.ndarray | None, eye_open_min: float) -> bool:
    """False when no face is present or either eye is closed (openness < eye_open_min).

    The extreme-head-pose check is added in M1.
    """
    if lms_px is None:
        return False
    return all(eye_openness(lms_px, e) >= eye_open_min for e in EYES)
