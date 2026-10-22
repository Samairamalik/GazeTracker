"""Check the hard-coded landmark indices against a real face: matplotlib's public-domain
grace_hopper.jpg sample image (unmirrored, eyes open). Skipped if the model file is missing."""
from pathlib import Path

import cv2
import matplotlib.cbook as cbook
import numpy as np
import pytest

from gazeread.features.landmarks import (
    EYES, LEFT_IRIS_CENTER, LEFT_IRIS_RING, NUM_LANDMARKS, RIGHT_IRIS_CENTER, RIGHT_IRIS_RING,
    LandmarkDetector, is_valid,
)

MODEL = Path(__file__).resolve().parents[1] / "models/face_landmarker.task"
NOSE_BRIDGE = 168
IRISES = {"right": (RIGHT_IRIS_CENTER, RIGHT_IRIS_RING), "left": (LEFT_IRIS_CENTER, LEFT_IRIS_RING)}


@pytest.fixture(scope="module")
def lms() -> np.ndarray:
    if not MODEL.exists():
        pytest.skip("models/face_landmarker.task not downloaded")
    img = cv2.imread(cbook.get_sample_data("grace_hopper.jpg", asfileobj=False))
    det = LandmarkDetector(str(MODEL))
    d = det.detect(img, 0.0)
    det.close()
    assert d is not None
    return d.lms_px


def test_landmark_count(lms):
    assert lms.shape == (NUM_LANDMARKS, 3)


@pytest.mark.parametrize("eye", ["right", "left"])
def test_eye_indices(lms, eye):
    center, ring = IRISES[eye]
    outer, inner, upper, lower = EYES[eye]
    radius = np.mean(np.linalg.norm(lms[list(ring), :2] - lms[center, :2], axis=1))
    assert np.linalg.norm(lms[list(ring), :2].mean(0) - lms[center, :2]) < 0.25 * radius
    assert abs(lms[outer, 0] - lms[NOSE_BRIDGE, 0]) > abs(lms[inner, 0] - lms[NOSE_BRIDGE, 0])
    assert lms[upper, 1] < lms[center, 1] < lms[lower, 1]
    assert min(lms[outer, 0], lms[inner, 0]) < lms[center, 0] < max(lms[outer, 0], lms[inner, 0])


def test_subject_right_eye_on_image_left(lms):
    assert lms[RIGHT_IRIS_CENTER, 0] < lms[LEFT_IRIS_CENTER, 0]


def test_open_eyes_valid(lms):
    assert is_valid(lms, eye_open_min=0.15)
