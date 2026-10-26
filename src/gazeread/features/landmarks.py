"""Thin wrapper around MediaPipe FaceLandmarker (Tasks API, mediapipe 1.1.1),
with automatic OpenCV Face + Iris pupil-centroid fallback when mediapipe is unavailable.

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
import logging
import math

import cv2
import numpy as np

log = logging.getLogger("gazeread")

# Landmark indices. Verified 2026-10-22 on matplotlib's public-domain grace_hopper.jpg:
# geometric checks (tests/test_landmark_indices.py) plus a visual check of a labeled crop.
RIGHT_IRIS_CENTER, RIGHT_IRIS_RING = 468, (469, 470, 471, 472)
LEFT_IRIS_CENTER, LEFT_IRIS_RING = 473, (474, 475, 476, 477)
EYES = {
    # eye: (outer corner, inner corner, upper lid, lower lid)
    "right": (33, 133, 159, 145),
    "left": (263, 362, 386, 374),
}
NUM_LANDMARKS = 478

# Key 3D wireframe contour chains (MediaPipe canonical indices) for the privacy-preserving Biometric HUD.
HUD_CONTOURS = {
    "oval": (10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
             400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
             54, 103, 67, 109, 10),
    "right_eye": (33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7, 33),
    "left_eye": (263, 466, 388, 387, 386, 385, 384, 398, 362, 382, 381, 380, 374, 373, 390, 249, 263),
    "right_brow": (46, 53, 52, 65, 55, 70, 63, 105, 66, 107),
    "left_brow": (276, 283, 282, 295, 285, 300, 293, 334, 296, 336),
    "nose": (168, 6, 197, 195, 5, 4, 1, 19, 94, 2),
    "lips": (61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61),
}


@dataclass
class Detection:
    lms_px: np.ndarray       # (478, 3) landmarks; x, y in image px, z scaled by image width
    transform: np.ndarray    # (4, 4) canonical-face -> detected-face; translation in cm
    frame_shape: tuple[int, int] = (480, 640)


class _OpenCVFallbackTracker:
    """Real-webcam OpenCV Face + Dark-Pupil/Iris centroid tracker used when mediapipe is not installed."""

    def __init__(self):
        haar_dir = getattr(cv2.data, "haarcascades", "")
        self._face_cas = cv2.CascadeClassifier(haar_dir + "haarcascade_frontalface_default.xml")
        self._smooth_box = None

    @staticmethod
    def _find_iris_centroid(gray_roi: np.ndarray) -> tuple[float, float, float]:
        """Return (cx, cy, openness) inside an eye ROI using intensity-weighted dark pupil centroid."""
        rh, rw = gray_roi.shape[:2]
        if rh < 6 or rw < 10:
            return rw / 2.0, rh / 2.0, 0.30
        sub = gray_roi[int(rh * 0.18):int(rh * 0.85), int(rw * 0.12):int(rw * 0.88)]
        sh, sw = sub.shape[:2]
        if sh < 4 or sw < 4:
            return rw / 2.0, rh / 2.0, 0.30
        blur = cv2.GaussianBlur(sub, (5, 5), 0)
        thresh = np.percentile(blur, 18)
        weights = np.clip(thresh - blur.astype(float) + 1.0, 0.0, None) ** 2
        total = weights.sum()
        if total <= 1e-6:
            iy, ix = np.unravel_index(int(np.argmin(blur)), blur.shape)
        else:
            ys, xs = np.indices(blur.shape)
            ix = float((xs * weights).sum() / total)
            iy = float((ys * weights).sum() / total)
        cx = int(rw * 0.12) + ix
        cy = int(rh * 0.18) + iy
        # Estimate eyelid openness from vertical gradient energy in the eye ROI
        gy = np.abs(cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)).mean()
        openness = float(np.clip(0.12 + 0.008 * gy, 0.06, 0.36))
        return float(cx), float(cy), openness

    def detect(self, frame_bgr: np.ndarray) -> Detection | None:
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._face_cas.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(80, 80))
        if len(faces) == 0:
            return None
        # Pick largest face
        fx, fy, fw, fh = max(faces, key=lambda b: b[2] * b[3])
        box = np.array([fx, fy, fw, fh], float)
        self._smooth_box = box if self._smooth_box is None else (0.65 * box + 0.35 * self._smooth_box)
        fx, fy, fw, fh = self._smooth_box

        lms = np.zeros((NUM_LANDMARKS, 3), float)
        fcx, fcy = fx + fw * 0.5, fy + fh * 0.5

        # Right eye (subject's right = image left) and Left eye (subject's left = image right)
        r_rx0, r_ry0, r_rw, r_rh = int(fx + fw * 0.16), int(fy + fh * 0.26), int(fw * 0.28), int(fh * 0.20)
        l_rx0, l_ry0, l_rw, l_rh = int(fx + fw * 0.56), int(fy + fh * 0.26), int(fw * 0.28), int(fh * 0.20)

        r_roi = gray[max(0, r_ry0):min(h, r_ry0 + r_rh), max(0, r_rx0):min(w, r_rx0 + r_rw)]
        l_roi = gray[max(0, l_ry0):min(h, l_ry0 + l_rh), max(0, l_rx0):min(w, l_rx0 + l_rw)]
        rcx, rcy, r_open = self._find_iris_centroid(r_roi)
        lcx, lcy, l_open = self._find_iris_centroid(l_roi)

        # Populate EYES landmarks so iris_uv and eye_openness work identically to MediaPipe
        # Right eye: outer=33 (leftmost), inner=133 (rightmost)
        r_out = np.array([fx + fw * 0.18, fy + fh * 0.36])
        r_in = np.array([fx + fw * 0.42, fy + fh * 0.36])
        r_w = np.linalg.norm(r_in - r_out)
        r_mid = 0.5 * (r_out + r_in)
        lms[33, :2] = r_out
        lms[133, :2] = r_in
        lms[159, :2] = [r_mid[0], r_mid[1] - 0.5 * r_open * r_w]
        lms[145, :2] = [r_mid[0], r_mid[1] + 0.5 * r_open * r_w]
        r_iris_xy = np.array([r_rx0 + rcx, r_ry0 + rcy])
        lms[RIGHT_IRIS_CENTER, :2] = r_iris_xy
        for k, idx in enumerate(RIGHT_IRIS_RING):
            ang = k * math.pi / 2
            lms[idx, :2] = r_iris_xy + np.array([math.cos(ang), math.sin(ang)]) * (0.12 * r_w)

        # Left eye: outer=263 (rightmost), inner=362 (leftmost)
        l_in = np.array([fx + fw * 0.58, fy + fh * 0.36])
        l_out = np.array([fx + fw * 0.82, fy + fh * 0.36])
        l_w = np.linalg.norm(l_out - l_in)
        l_mid = 0.5 * (l_out + l_in)
        lms[263, :2] = l_out
        lms[362, :2] = l_in
        lms[386, :2] = [l_mid[0], l_mid[1] - 0.5 * l_open * l_w]
        lms[374, :2] = [l_mid[0], l_mid[1] + 0.5 * l_open * l_w]
        l_iris_xy = np.array([l_rx0 + lcx, l_ry0 + lcy])
        lms[LEFT_IRIS_CENTER, :2] = l_iris_xy
        for k, idx in enumerate(LEFT_IRIS_RING):
            ang = k * math.pi / 2
            lms[idx, :2] = l_iris_xy + np.array([math.cos(ang), math.sin(ang)]) * (0.12 * l_w)

        # Populate HUD contour landmarks around the detected face so the 3D Cyber-Mesh HUD animates live
        for k, idx in enumerate(HUD_CONTOURS["oval"]):
            a = 2 * math.pi * k / max(1, len(HUD_CONTOURS["oval"]) - 1) - math.pi / 2
            lms[idx, :2] = [fcx + 0.46 * fw * math.cos(a), fcy + 0.54 * fh * math.sin(a)]
        for k, idx in enumerate(HUD_CONTOURS["right_eye"]):
            a = 2 * math.pi * k / max(1, len(HUD_CONTOURS["right_eye"]) - 1)
            lms[idx, :2] = [r_mid[0] + 0.5 * r_w * math.cos(a), r_mid[1] + 0.5 * r_open * r_w * math.sin(a)]
        for k, idx in enumerate(HUD_CONTOURS["left_eye"]):
            a = 2 * math.pi * k / max(1, len(HUD_CONTOURS["left_eye"]) - 1)
            lms[idx, :2] = [l_mid[0] + 0.5 * l_w * math.cos(a), l_mid[1] + 0.5 * l_open * l_w * math.sin(a)]
        for k, idx in enumerate(HUD_CONTOURS["right_brow"]):
            lms[idx, :2] = [r_out[0] + (r_in[0] - r_out[0]) * (k / 9.0), fy + fh * 0.24]
        for k, idx in enumerate(HUD_CONTOURS["left_brow"]):
            lms[idx, :2] = [l_in[0] + (l_out[0] - l_in[0]) * (k / 9.0), fy + fh * 0.24]
        for k, idx in enumerate(HUD_CONTOURS["nose"]):
            lms[idx, :2] = [fcx, fy + fh * (0.34 + 0.045 * k)]
        for k, idx in enumerate(HUD_CONTOURS["lips"]):
            a = 2 * math.pi * k / max(1, len(HUD_CONTOURS["lips"]) - 1)
            lms[idx, :2] = [fcx + 0.18 * fw * math.cos(a), fy + fh * 0.76 + 0.045 * fh * math.sin(a)]

        # Approximate 4x4 head pose transform from face box position and size
        tz_cm = float(np.clip(11500.0 / max(fw, 40.0), 35.0, 95.0))
        tx_cm = float((fcx / w - 0.5) * 30.0)
        ty_cm = float((fcy / h - 0.5) * 22.0)
        yaw_rad = math.radians((fcx / w - 0.5) * -28.0)
        pitch_rad = math.radians((fcy / h - 0.5) * 20.0)
        cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
        cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
        T = np.eye(4)
        T[:3, :3] = np.array([[cy, 0, sy], [sp * sy, cp, -sp * cy], [-cp * sy, sp, cp * cy]])
        T[:3, 3] = [tx_cm, ty_cm, tz_cm]
        return Detection(lms_px=lms, transform=T, frame_shape=(h, w))


class LandmarkDetector:
    """FaceLandmarker in VIDEO mode, with automatic OpenCV fallback when mediapipe is unavailable."""

    def __init__(self, model_path: str, num_faces: int = 1, video: bool = True):
        """video=True for a live stream (temporal tracking); False for unrelated still images."""
        self._video = video
        self._last_ms = -1
        self._landmarker = None
        self._fallback = None
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions

            self._mp = mp
            options = vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
                num_faces=num_faces,
                output_facial_transformation_matrixes=True,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            log.warning("MediaPipe unavailable (%s); using built-in OpenCV Face+Iris tracker.", e)
            self._fallback = _OpenCVFallbackTracker()

    def detect(self, frame_bgr: np.ndarray, t_s: float = 0.0) -> Detection | None:
        """Run on one BGR frame captured at monotonic time `t_s` (seconds; ignored for still
        images). None if no face."""
        if self._fallback is not None:
            return self._fallback.detect(frame_bgr)
        mp = self._mp
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self._video:
            t_ms = max(int(t_s * 1000), self._last_ms + 1)   # MediaPipe needs strictly increasing ms
            self._last_ms = t_ms
            result = self._landmarker.detect_for_video(image, t_ms)
        else:
            result = self._landmarker.detect(image)
        if not result.face_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        lms = np.array([(p.x * w, p.y * h, p.z * w) for p in result.face_landmarks[0]])
        return Detection(
            lms_px=lms,
            transform=np.array(result.facial_transformation_matrixes[0]),
            frame_shape=(h, w),
        )

    def close(self) -> None:
        if self._landmarker is not None:
            self._landmarker.close()


def iris_center_5pt(lms_px: np.ndarray, eye: str) -> np.ndarray:
    """5-point iris centroid (center + 4 ring landmarks) in image px, falling back to center if ring is unset."""
    c_idx = RIGHT_IRIS_CENTER if eye == "right" else LEFT_IRIS_CENTER
    r_idx = RIGHT_IRIS_RING if eye == "right" else LEFT_IRIS_RING
    ring_pts = lms_px[list(r_idx), :2]
    if np.any(ring_pts != 0.0):
        pts = np.vstack([lms_px[c_idx, :2], ring_pts])
        return pts.mean(axis=0)
    return lms_px[c_idx, :2]


def extract_hud_mesh(det: Detection | None) -> dict | None:
    """Extract normalized [0..1] 3D wireframe contours and iris vectors for the Biometric HUD."""
    if det is None:
        return None
    lms = det.lms_px
    h, w = det.frame_shape if det.frame_shape else (480, 640)
    w = max(float(w), 1.0)
    h = max(float(h), 1.0)

    def norm_pts(indices):
        return [[round(float(lms[idx, 0] / w), 4),
                 round(float(lms[idx, 1] / h), 4),
                 round(float(lms[idx, 2] / w), 4)] for idx in indices]

    r_iris = norm_pts((RIGHT_IRIS_CENTER, *RIGHT_IRIS_RING))
    l_iris = norm_pts((LEFT_IRIS_CENTER, *LEFT_IRIS_RING))
    return {
        "contours": {name: norm_pts(idxs) for name, idxs in HUD_CONTOURS.items()},
        "right_iris": r_iris,
        "left_iris": l_iris,
    }


def synthetic_hud_mesh(u_r: float, v_r: float, u_l: float, v_l: float,
                       open_r: float, open_l: float,
                       yaw: float = 0.0, pitch: float = 0.0, roll: float = 0.0) -> dict:
    """Procedural 3D cyber-mesh for synthetic mode so the Biometric HUD animates realistically."""
    yr, pr, rr = math.radians(yaw), math.radians(pitch), math.radians(roll)
    cy, sy = math.cos(yr), math.sin(yr)
    cp, sp = math.cos(pr), math.sin(pr)
    cr, sr = math.cos(rr), math.sin(rr)

    def proj(x: float, y: float, z: float) -> list[float]:
        # Rotate Y (yaw), X (pitch), Z (roll), then center in [0..1]
        x1, z1 = x * cy + z * sy, -x * sy + z * cy
        y2, z2 = y * cp - z1 * sp, y * sp + z1 * cp
        x3, y3 = x1 * cr - y2 * sr, x1 * sr + y2 * cr
        return [round(0.5 + x3, 4), round(0.5 + y3, 4), round(z2, 4)]

    oval = []
    for k in range(25):
        ang = 2 * math.pi * k / 24 - math.pi / 2
        rx = 0.22 * math.cos(ang) * (1.0 - 0.12 * max(0.0, math.sin(ang)))
        ry = 0.29 * math.sin(ang)
        rz = -0.08 * (1.0 - math.cos(ang) ** 2)
        oval.append(proj(rx, ry, rz))

    def eye_loop(cx: float, cy_off: float, opn: float):
        pts = []
        lid_h = max(0.004, opn * 0.095)
        for k in range(13):
            a = 2 * math.pi * k / 12
            pts.append(proj(cx + 0.055 * math.cos(a), cy_off + lid_h * math.sin(a), -0.09))
        return pts

    def iris_pts(cx: float, cy_off: float, u: float, v: float, sign_u: float):
        ix = cx + sign_u * u * 0.11
        iy = cy_off + v * 0.11
        center = proj(ix, iy, -0.095)
        ring = [proj(ix + 0.018 * math.cos(a), iy + 0.022 * math.sin(a), -0.095)
                for a in (0, math.pi / 2, math.pi, 3 * math.pi / 2)]
        return [center, *ring]

    r_brow = [proj(-0.145 + 0.095 * (k / 5), -0.115 - 0.018 * math.sin(math.pi * k / 5), -0.085) for k in range(6)]
    l_brow = [proj(0.05 + 0.095 * (k / 5), -0.115 - 0.018 * math.sin(math.pi * k / 5), -0.085) for k in range(6)]
    nose = [proj(0.0, -0.08 + 0.03 * k, -0.11 - 0.01 * k) for k in range(6)]
    lips = [proj(0.075 * math.cos(2 * math.pi * k / 12), 0.145 + 0.024 * math.sin(2 * math.pi * k / 12), -0.09)
            for k in range(13)]

    return {
        "contours": {
            "oval": oval,
            "right_eye": eye_loop(-0.095, -0.04, open_r),
            "left_eye": eye_loop(0.095, -0.04, open_l),
            "right_brow": r_brow,
            "left_brow": l_brow,
            "nose": nose,
            "lips": lips,
        },
        "right_iris": iris_pts(-0.095, -0.04, u_r, v_r, -1.0),
        "left_iris": iris_pts(0.095, -0.04, u_l, v_l, 1.0),
    }


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
