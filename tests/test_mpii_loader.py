from pathlib import Path

import cv2
import numpy as np
import pytest
import scipy.io as sio

from gazeread.data.mpiifacegaze import load_participant, mm_to_px, parse_annotation, px_to_mm

LINE = ("day01/0005.jpg 476 758 594 366 637 365 719 366 762 368 626 495 726 496 -0.232179 0.055685 "
        "0.018205 28.351504 1.174807 529.783734 27.792112 23.422692 524.537075 11.040978 166.869249 "
        "-27.728178 right")


def make_fixture(root: Path, pid="p99"):
    """A one-person dataset in the verified MPIIFaceGaze layout (2 annotation lines)."""
    d = root / pid
    (d / "Calibration").mkdir(parents=True)
    second = LINE.replace("day01/0005.jpg 476 758", "day01/0030.jpg 1078 693")
    (d / f"{pid}.txt").write_text(LINE + "\n" + second + "\n")
    sio.savemat(d / "Calibration" / "screenSize.mat",
                {"width_pixel": 1280.0, "height_pixel": 800.0, "width_mm": 286.47, "height_mm": 179.04})
    rv = np.array([[0.0], [np.pi], [0.0]])                 # screen faces the camera
    sio.savemat(d / "Calibration" / "monitorPose.mat", {"rvects": rv, "tvecs": np.array([[120.0], [-5.0], [-30.0]])})
    return root


def test_parse_annotation_fields():
    tmp = Path(pytest.importorskip("tempfile").mkdtemp()) / "p.txt"
    tmp.write_text(LINE + "\n")
    names, tgt, gt = parse_annotation(tmp)
    assert names == ["day01/0005.jpg"]
    np.testing.assert_allclose(tgt, [[476, 758]])
    np.testing.assert_allclose(gt, [[11.040978, 166.869249, -27.728178]])


def test_bad_line_rejected(tmp_path):
    (tmp_path / "p.txt").write_text("day01/0005.jpg 1 2 3\n")
    with pytest.raises(ValueError):
        parse_annotation(tmp_path / "p.txt")


def test_px_mm_round_trip():
    mpp = np.array([286.47 / 1280, 179.04 / 800])
    px = np.array([[0, 0], [640, 400], [1280, 800]], float)
    np.testing.assert_allclose(mm_to_px(px_to_mm(px, mpp), mpp), px)
    np.testing.assert_allclose(px_to_mm(px, mpp)[-1], [286.47, 179.04])


def test_load_participant_camera_relative_targets(tmp_path):
    P = load_participant(make_fixture(tmp_path), "p99")
    assert len(P.image_paths) == 2 and P.image_paths[0].name == "0005.jpg"
    R, _ = cv2.Rodrigues(np.array([0.0, np.pi, 0.0]))
    cam = (-R.T @ np.array([120.0, -5.0, -30.0]))[:2]
    np.testing.assert_allclose(P.camera_on_screen_mm, cam, atol=1e-9)
    np.testing.assert_allclose(P.target_mm()[0], np.array([476, 758]) * P.mm_per_px - cam)
    np.testing.assert_allclose(P.target_frac()[1], [1078 / 1280, 693 / 800])
    # distances between targets are screen-plane mm, unaffected by the camera origin
    d_mm = np.linalg.norm(P.target_mm()[0] - P.target_mm()[1])
    assert np.isclose(d_mm, np.linalg.norm((P.target_px[0] - P.target_px[1]) * P.mm_per_px))


REAL = Path(__file__).resolve().parents[1] / "data/raw/MPIIFaceGaze"


@pytest.mark.skipif(not REAL.exists(), reason="MPIIFaceGaze not downloaded")
def test_real_dataset_counts_and_geometry():
    total = 0
    for i in range(15):
        P = load_participant(REAL, f"p{i:02d}")
        total += len(P.image_paths)
        cam = (P.R @ np.c_[P.target_px * P.mm_per_px, np.zeros(len(P.target_px))].T).T + P.t
        assert np.abs(cam - P.gt_cam).max() < 1e-3          # screen geometry reproduces the 3D targets
    assert total == 37667
