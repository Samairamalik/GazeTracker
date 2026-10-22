"""Method B: L2CS-Net gaze angles from a MediaPipe face crop.

FrameFeatures.vec = [gaze_yaw, gaze_pitch, head_yaw, head_pitch], all degrees. Gaze angles
come from L2CS-Net (Gaze360 weights: 90 bins of 4 deg, angle = E[bin] * 4 - 180); head pose
comes from the same MediaPipe pass, so "B+pose" can use it without a second model.
Preprocessing copies upstream: crop -> RGB -> resize 224x224 -> Resize(448) -> ImageNet norm.
(Upstream train.py and test.py both use Resize(448); Py-Feat's wrapper feeds 224 instead, we follow upstream.)
Validity is decided by the caller (same gate as Method A).
"""
import numpy as np
import torch
from torchvision import transforms

import cv2

from gazeread.features.head_pose import pose_from_matrix
from gazeread.features.landmarks import Detection
from gazeread.gaze.l2cs_model import load_l2cs
from gazeread.types import FrameFeatures

L2CS_NAMES = ("gaze_yaw", "gaze_pitch", "head_yaw", "head_pitch")
L2CS_SETS = {"gaze": [0, 1], "gaze_pose": [0, 1, 2, 3]}
BIN_DEG, ANGLE_OFFSET = 4.0, -180.0

_TRANSFORM = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(448),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def decode_bins(logits: torch.Tensor) -> np.ndarray:
    """(n, 90) logits -> (n,) angles in degrees (softmax expectation over bin index)."""
    p = torch.softmax(logits, dim=1)
    idx = torch.arange(logits.shape[1], dtype=p.dtype, device=p.device)
    return ((p * idx).sum(dim=1) * BIN_DEG + ANGLE_OFFSET).cpu().numpy()


def face_crop_box(lms_px: np.ndarray, scale: float, frame_shape) -> tuple[int, int, int, int]:
    """Square box (x0, y0, x1, y1) around the landmarks, side = scale * max(w, h), clipped to the frame."""
    h, w = frame_shape[:2]
    x0, y0 = lms_px[:, 0].min(), lms_px[:, 1].min()
    x1, y1 = lms_px[:, 0].max(), lms_px[:, 1].max()
    cx, cy, half = (x0 + x1) / 2, (y0 + y1) / 2, scale * max(x1 - x0, y1 - y0) / 2
    return (int(max(0, cx - half)), int(max(0, cy - half)), int(min(w, cx + half)), int(min(h, cy + half)))


def preprocess(frame_bgr: np.ndarray, box) -> torch.Tensor:
    x0, y0, x1, y1 = box
    face = cv2.resize(cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2RGB), (224, 224))
    return _TRANSFORM(face).unsqueeze(0)


def pick_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(pref)


class L2CSBackend:
    def __init__(self, weights: str, crop_scale: float = 1.2, device: str = "auto"):
        self.device = pick_device(device)
        self.model = load_l2cs(weights, self.device)
        self.crop_scale = crop_scale

    @torch.inference_mode()
    def gaze_angles(self, frame_bgr: np.ndarray, lms_px: np.ndarray) -> tuple[float, float]:
        """(yaw, pitch) in degrees for the face around `lms_px`."""
        x = preprocess(frame_bgr, face_crop_box(lms_px, self.crop_scale, frame_bgr.shape)).to(self.device)
        yaw_logits, pitch_logits = self.model(x)
        return float(decode_bins(yaw_logits)[0]), float(decode_bins(pitch_logits)[0])

    def from_detection(self, frame_bgr: np.ndarray, det: Detection | None, valid: bool, t: float) -> FrameFeatures:
        if det is None:
            return FrameFeatures(t=t, valid=False, vec=np.full(len(L2CS_NAMES), np.nan))
        yaw, pitch = self.gaze_angles(frame_bgr, det.lms_px)
        head_yaw, head_pitch = pose_from_matrix(det.transform)[:2]
        return FrameFeatures(t=t, valid=valid, vec=np.array([yaw, pitch, head_yaw, head_pitch]))
