import numpy as np

from gazeread.features.landmarks import EYES, NUM_LANDMARKS, eye_openness, is_valid


def make_face(right_open: float, left_open: float, width: float = 40.0) -> np.ndarray:
    """Hand-made landmarks: each eye `width` px wide with lid gap = openness * width."""
    lms = np.zeros((NUM_LANDMARKS, 3))
    for (eye, (outer, inner, upper, lower)), x0, op in zip(
        EYES.items(), (100.0, 200.0), (right_open, left_open)
    ):
        lms[outer, :2] = (x0, 50.0)
        lms[inner, :2] = (x0 + width, 50.0)
        lms[upper, :2] = (x0 + width / 2, 50.0 - op * width / 2)
        lms[lower, :2] = (x0 + width / 2, 50.0 + op * width / 2)
    return lms


def test_openness_recovers_ratio():
    lms = make_face(0.3, 0.25)
    assert np.isclose(eye_openness(lms, "right"), 0.3)
    assert np.isclose(eye_openness(lms, "left"), 0.25)


def test_open_eyes_valid():
    assert is_valid(make_face(0.3, 0.3), eye_open_min=0.15)


def test_one_eye_closed_invalid():
    assert not is_valid(make_face(0.3, 0.05), eye_open_min=0.15)
    assert not is_valid(make_face(0.05, 0.3), eye_open_min=0.15)


def test_no_face_invalid():
    assert not is_valid(None, eye_open_min=0.15)


def test_degenerate_eye_width_is_closed():
    assert eye_openness(make_face(0.3, 0.3, width=0.0), "right") == 0.0
