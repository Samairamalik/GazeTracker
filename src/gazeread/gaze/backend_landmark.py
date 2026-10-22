"""Method A feature extractor: MediaPipe landmarks -> fixed-length FrameFeatures.vec.

Vector order (FEATURE_NAMES), units in brackets:
    0 uR, 1 vR      right-eye iris offset [eye widths]
    2 uL, 3 vL      left-eye iris offset  [eye widths]
    4 openR, 5 openL  eye openness [lid distance / eye width]
    6 yaw, 7 pitch, 8 roll  head rotation [deg]
    9 tx, 10 ty, 11 tz      head translation [cm]
"Right"/"left" = the subject's eyes.

u = (iris centre - eye centre) projected on the inner->outer corner axis, / eye width (spec 4.1).
v = the perpendicular component / eye width, with the perpendicular oriented image-down for
both eyes, so v > 0 means the iris sits lower in the image for either eye. Because the
inner->outer axes point in opposite image directions, u has opposite signs for the two eyes
when the gaze moves sideways.
"""
import numpy as np

from gazeread.features.head_pose import pose_from_matrix
from gazeread.features.landmarks import (
    EYES, LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER, Detection, LandmarkDetector, eye_openness, is_valid,
)
from gazeread.types import FrameFeatures

FEATURE_NAMES = ("uR", "vR", "uL", "vL", "openR", "openL",
                 "yaw", "pitch", "roll", "tx", "ty", "tz")
# Column subsets used by the calibration map (configs: calib.features).
FEATURE_SETS = {"iris": [0, 1, 2, 3], "iris_pose": [0, 1, 2, 3, 6, 7],
                "iris_open": [0, 1, 2, 3, 4, 5]}   # openness: lids drop when looking down
IRIS_CENTERS = {"right": RIGHT_IRIS_CENTER, "left": LEFT_IRIS_CENTER}


def iris_uv(lms_px: np.ndarray, eye: str) -> tuple[float, float]:
    """Iris offset (u, v) in eye widths for `eye` ("right"/"left"), from landmarks in image px."""
    outer, inner, _, _ = EYES[eye]
    o, i = lms_px[outer, :2], lms_px[inner, :2]
    width = float(np.linalg.norm(o - i))
    if width <= 0:
        return 0.0, 0.0
    axis = (o - i) / width
    perp = np.array([-axis[1], axis[0]])
    if perp[1] < 0:
        perp = -perp
    d = lms_px[IRIS_CENTERS[eye], :2] - (o + i) / 2
    return float(d @ axis / width), float(d @ perp / width)


def features_from_detection(det: Detection | None, eye_open_min: float,
                            max_yaw_deg: float, max_pitch_deg: float) -> tuple[np.ndarray, bool]:
    """Feature vector (NaN if no face) and validity (face, both eyes open, pose within limits)."""
    if det is None:
        return np.full(len(FEATURE_NAMES), np.nan), False
    lms = det.lms_px
    uR, vR = iris_uv(lms, "right")
    uL, vL = iris_uv(lms, "left")
    yaw, pitch, roll, tx, ty, tz = pose_from_matrix(det.transform)
    vec = np.array([uR, vR, uL, vL, eye_openness(lms, "right"), eye_openness(lms, "left"),
                    yaw, pitch, roll, tx, ty, tz])
    valid = (is_valid(lms, eye_open_min)
             and abs(yaw) <= max_yaw_deg and abs(pitch) <= max_pitch_deg)
    return vec, valid


class LandmarkBackend:
    """Frame -> FrameFeatures using MediaPipe FaceLandmarker (VIDEO mode)."""

    def __init__(self, model_path: str, eye_open_min: float, max_yaw_deg: float,
                 max_pitch_deg: float, num_faces: int = 1):
        self._det = LandmarkDetector(model_path, num_faces)
        self._limits = (eye_open_min, max_yaw_deg, max_pitch_deg)

    def detect(self, frame_bgr: np.ndarray, t: float) -> Detection | None:
        """MediaPipe pass, shared with Method B. `t` is the capture time in seconds (monotonic)."""
        return self._det.detect(frame_bgr, t)

    def from_detection(self, det: Detection | None, t: float) -> FrameFeatures:
        vec, valid = features_from_detection(det, *self._limits)
        return FrameFeatures(t=t, valid=valid, vec=vec)

    def process(self, frame_bgr: np.ndarray, t: float) -> FrameFeatures:
        """`t` is the capture time in seconds (monotonic)."""
        return self.from_detection(self.detect(frame_bgr, t), t)

    def close(self) -> None:
        self._det.close()
