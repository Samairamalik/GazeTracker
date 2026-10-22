import math

import numpy as np

from gazeread.features.head_pose import pose_from_matrix
from gazeread.features.landmarks import Detection, EYES, NUM_LANDMARKS
from gazeread.gaze.backend_landmark import (
    FEATURE_NAMES, IRIS_CENTERS, features_from_detection, iris_uv,
)

# Unmirrored image: subject's right eye on the image left, outer corner further left.
EYE_GEOM = {"right": ((100.0, 50.0), (140.0, 50.0)), "left": ((240.0, 50.0), (200.0, 50.0))}


def make_face(iris_offset=(0.0, 0.0), open_ratio=0.3) -> np.ndarray:
    lms = np.zeros((NUM_LANDMARKS, 3))
    for eye, (outer_xy, inner_xy) in EYE_GEOM.items():
        outer, inner, upper, lower = EYES[eye]
        lms[outer, :2], lms[inner, :2] = outer_xy, inner_xy
        cx = (outer_xy[0] + inner_xy[0]) / 2
        lms[upper, :2] = (cx, 50.0 - open_ratio * 20)
        lms[lower, :2] = (cx, 50.0 + open_ratio * 20)
        lms[IRIS_CENTERS[eye], :2] = (cx + iris_offset[0], 50.0 + iris_offset[1])
    return lms


def rot_y(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    T = np.eye(4)
    T[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    return T


def test_centered_iris_is_zero():
    lms = make_face()
    for eye in EYES:
        np.testing.assert_allclose(iris_uv(lms, eye), (0.0, 0.0), atol=1e-12)


def test_iris_moving_image_right():
    lms = make_face((4.0, 0.0))
    uR, vR = iris_uv(lms, "right")      # inner->outer points image-left for the right eye
    uL, vL = iris_uv(lms, "left")       # and image-right for the left eye
    assert np.isclose(uR, -0.1) and np.isclose(uL, 0.1)
    assert np.isclose(vR, 0.0) and np.isclose(vL, 0.0)


def test_iris_moving_image_down_same_sign_both_eyes():
    lms = make_face((0.0, 4.0))
    assert np.isclose(iris_uv(lms, "right")[1], 0.1)
    assert np.isclose(iris_uv(lms, "left")[1], 0.1)


def test_feature_vector_layout_and_validity():
    vec, valid = features_from_detection(Detection(make_face(), np.eye(4)), 0.15, 30, 25)
    assert vec.shape == (len(FEATURE_NAMES),)
    assert valid
    assert np.isclose(vec[4], 0.3) and np.isclose(vec[5], 0.3)


def test_extreme_yaw_is_invalid():
    _, valid = features_from_detection(Detection(make_face(), rot_y(40)), 0.15, 30, 25)
    assert not valid


def test_pose_from_matrix_yaw():
    yaw, pitch, roll, *_ = pose_from_matrix(rot_y(20))
    assert np.isclose(yaw, 20) and np.isclose(pitch, 0) and np.isclose(roll, 0)


def test_no_face():
    vec, valid = features_from_detection(None, 0.15, 30, 25)
    assert not valid and np.isnan(vec).all()


def test_closed_eye_is_invalid():
    _, valid = features_from_detection(Detection(make_face(open_ratio=0.05), np.eye(4)), 0.15, 30, 25)
    assert not valid


def test_iris_open_feature_set_includes_openness():
    from gazeread.gaze.backend_landmark import FEATURE_SETS
    assert [FEATURE_NAMES[i] for i in FEATURE_SETS["iris_open"]] == ["uR", "vR", "uL", "vL", "openR", "openL"]
