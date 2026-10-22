"""Head pose from MediaPipe's 4x4 facial transformation matrix (canonical face -> detected face)."""
import math

import numpy as np


def pose_from_matrix(transform: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """Return (yaw, pitch, roll) in degrees and (tx, ty, tz) in cm.

    Angles are a ZYX Euler decomposition of the rotation block; signs follow MediaPipe's
    camera frame and only need to be consistent (they feed a regression and a magnitude gate).
    """
    r = transform[:3, :3]
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2])))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    tx, ty, tz = (float(v) for v in transform[:3, 3])
    return yaw, pitch, roll, tx, ty, tz
