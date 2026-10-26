import math
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np

from gazeread.app.server import STATIC, SyntheticSource, list_passages, wait_until_logged
from gazeread.features.landmarks import (
    Detection, NUM_LANDMARKS, RIGHT_IRIS_CENTER, RIGHT_IRIS_RING,
    extract_hud_mesh, iris_center_5pt,
)
from gazeread.gaze.backend_landmark import FEATURE_NAMES, FEATURE_SETS


def test_wait_until_logged_waits_for_in_flight_frame():
    p = SimpleNamespace(logged_t=-math.inf)
    t_end = time.monotonic()

    def capture_thread():
        time.sleep(0.05)
        p.logged_t = t_end - 0.01        # a frame captured before the end is still being written
        time.sleep(0.05)
        p.logged_t = t_end + 0.02        # first frame captured after the end

    threading.Thread(target=capture_thread).start()
    t0 = time.monotonic()
    assert wait_until_logged(p, t_end, timeout_s=1.0)
    assert time.monotonic() - t0 >= 0.09


def test_wait_until_logged_times_out():
    p = SimpleNamespace(logged_t=0.0)
    assert not wait_until_logged(p, 10.0, timeout_s=0.05)


def test_synthetic_source_produces_features_and_hud_mesh():
    synth = SyntheticSource(noise_px=5.0, seed=42)
    frames, mesh = synth.read(1.0, (640.0, 400.0), 1280.0, 800.0)
    assert "landmark" in frames and "l2cs" in frames
    assert frames["landmark"].vec.shape == (len(FEATURE_NAMES),)
    assert "contours" in mesh and "right_iris" in mesh and "left_iris" in mesh
    assert len(mesh["right_iris"]) == 5 and len(mesh["left_iris"]) == 5


def test_iris_5pt_uses_ring_when_present():
    lms = np.zeros((NUM_LANDMARKS, 3))
    lms[RIGHT_IRIS_CENTER, :2] = (100.0, 50.0)
    # Without ring points -> falls back to center
    np.testing.assert_allclose(iris_center_5pt(lms, "right"), (100.0, 50.0))
    # With symmetric ring points shifted +5 in x -> 5-pt centroid shifts by +4
    for r in RIGHT_IRIS_RING:
        lms[r, :2] = (105.0, 50.0)
    np.testing.assert_allclose(iris_center_5pt(lms, "right"), (104.0, 50.0))


def test_extract_hud_mesh_normalizes_coordinates():
    lms = np.full((NUM_LANDMARKS, 3), 100.0)
    det = Detection(lms_px=lms, transform=np.eye(4), frame_shape=(200, 400))
    mesh = extract_hud_mesh(det)
    assert mesh is not None
    assert mesh["right_iris"][0][:2] == [0.25, 0.5]


def test_6dof_feature_sets_and_passages_list():
    assert "tx" in [FEATURE_NAMES[i] for i in FEATURE_SETS["iris_pose_6dof"]]
    assert len(FEATURE_SETS["full"]) == len(FEATURE_NAMES)
    passages = list_passages(Path(STATIC) / "texts")
    assert len(passages) >= 4
    assert any(p["id"] == "alice_passage1" and p["quiz_count"] > 0 for p in passages)
