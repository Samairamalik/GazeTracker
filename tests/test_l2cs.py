from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from gazeread.gaze.backend_l2cs import decode_bins, face_crop_box, preprocess
from gazeread.gaze.calibration import select_model
from gazeread.gaze.l2cs_model import build_l2cs, load_l2cs

WEIGHTS = Path(__file__).resolve().parents[1] / "models/l2cs_gaze360_resnet50.safetensors"
WEIGHTS_SHA256 = "75405bdc01f7086b3887280fe50c28a14bc62463a53fa749781e4a3cf98eca0e"


def spike(bin_idx: int, n: int = 90) -> torch.Tensor:
    z = torch.full((1, n), -50.0)
    z[0, bin_idx] = 50.0
    return z


def test_decode_bins():
    assert np.isclose(decode_bins(spike(45))[0], 0.0)          # 45 * 4 - 180
    assert np.isclose(decode_bins(spike(50))[0], 20.0)
    assert np.isclose(decode_bins(torch.zeros(1, 90))[0], 44.5 * 4 - 180)   # uniform -> mean bin


def test_face_crop_box_square_scaled_and_clipped():
    lms = np.array([[100.0, 100.0], [200.0, 150.0]])           # 100 x 50 px landmark extent
    assert face_crop_box(lms, 1.2, (480, 640)) == (90, 65, 210, 185)   # side 120, centred
    x0, y0, x1, y1 = face_crop_box(np.array([[0.0, 0.0], [100.0, 100.0]]), 1.5, (480, 640))
    assert (x0, y0) == (0, 0) and x1 == 125 and y1 == 125


def test_preprocess_shape():
    x = preprocess(np.zeros((480, 640, 3), np.uint8), (100, 100, 300, 300))
    assert tuple(x.shape) == (1, 3, 448, 448)


def test_forward_returns_yaw_then_pitch():
    m = build_l2cs().eval()
    with torch.no_grad():
        for fc, b in ((m.fc_yaw_gaze, 60), (m.fc_pitch_gaze, 30)):
            fc.weight.zero_()
            fc.bias.fill_(-50.0)
            fc.bias[b] = 50.0
        yaw, pitch = m(torch.zeros(1, 3, 448, 448))
    assert yaw.shape == pitch.shape == (1, 90)
    assert np.isclose(decode_bins(yaw)[0], 60.0) and np.isclose(decode_bins(pitch)[0], -60.0)


def test_load_missing_weights_fails_clearly(tmp_path):
    with pytest.raises(RuntimeError, match="not found"):
        load_l2cs(tmp_path / "nope.pkl", torch.device("cpu"))


class NotAWeight:
    pass


def test_load_refuses_unsafe_pickle(tmp_path):
    p = tmp_path / "unsafe.pkl"
    torch.save({"obj": NotAWeight()}, p)
    with pytest.raises(RuntimeError, match="weights_only=True"):
        load_l2cs(p, torch.device("cpu"))


def test_load_plain_state_dict(tmp_path):
    p = tmp_path / "ok.pkl"
    torch.save(build_l2cs().state_dict(), p)
    assert load_l2cs(p, torch.device("cpu")).training is False


@pytest.mark.skipif(not WEIGHTS.exists(), reason="L2CS weights not downloaded")
def test_real_weights_load_and_predict_in_range():
    m = load_l2cs(WEIGHTS, torch.device("cpu"))
    x = preprocess(np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8), (200, 100, 440, 340))
    with torch.no_grad():
        yaw, pitch = m(x)
    assert -180 <= decode_bins(yaw)[0] <= 180 and -180 <= decode_bins(pitch)[0] <= 180


def test_calibration_on_yaw_pitch_features():
    rng = np.random.default_rng(4)
    X = rng.uniform(-20, 20, size=(9, 2))                      # degrees
    Y = X @ np.array([[30.0, 2.0], [-1.0, -25.0]]) + [756, 472]
    model, info = select_model(X, Y)
    assert info["degree"] == 1 and info["loo_rms_px"] < 1.0


def test_variants_from_config():
    from gazeread.app.server import make_variants
    cfg = yaml.safe_load(open("configs/default.yaml"))
    vs, primary = make_variants(cfg)
    assert [v.name for v in vs] == ["A"] and primary == "A"
    cfg["gaze"]["compare"] = True
    cfg["gaze"]["backend"] = "l2cs"
    cfg["l2cs"]["features"] = "gaze_pose"
    vs, primary = make_variants(cfg)
    assert [v.name for v in vs] == ["A", "B", "B+pose"] and primary == "B+pose"
    assert [v.sel for v in vs][1:] == [[0, 1], [0, 1, 2, 3]]


def test_load_safetensors_strict_and_key_mismatch(tmp_path):
    from safetensors.torch import save_file
    sd = build_l2cs().state_dict()
    good = tmp_path / "ok.safetensors"
    save_file({k: v.contiguous() for k, v in sd.items()}, str(good))
    assert load_l2cs(good, torch.device("cpu")).training is False
    bad = tmp_path / "bad.safetensors"
    save_file({("module." + k): v.contiguous() for k, v in sd.items()}, str(bad))
    with pytest.raises(RuntimeError, match="keys do not match"):
        load_l2cs(bad, torch.device("cpu"))


@pytest.mark.skipif(not WEIGHTS.exists(), reason="L2CS weights not downloaded")
def test_weights_file_hash():
    import hashlib
    assert hashlib.sha256(WEIGHTS.read_bytes()).hexdigest() == WEIGHTS_SHA256
