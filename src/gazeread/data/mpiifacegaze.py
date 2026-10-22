"""MPIIFaceGaze loader (spec 3.1).

Layout VERIFIED on 2026-10-22 against the downloaded MPIIFaceGaze.zip (942,343,290 bytes) and
its readme.txt:
- data/raw/MPIIFaceGaze/pXX/ for XX = 00..14 (15 participants), images in dayNN/ subfolders,
  JPEG 1280x720 full frames. 37,667 annotated images in total (p00 2927 ... p14 1500).
- pXX/pXX.txt: one line per image, 28 space-separated fields:
    1      image path relative to pXX/ (e.g. day01/0005.jpg)
    2-3    gaze target on the screen, pixels (x, y), origin top-left
    4-15   six facial landmarks (x, y) in image px: four eye corners, two mouth corners
    16-21  3D head pose (rotation vector, translation) in camera coordinates
    22-24  face centre "fc" in camera coordinates (mm)
    25-27  3D gaze target "gt" in camera coordinates (mm)
    28     "left"/"right": eye used for the MPIIGaze evaluation subset
- pXX/Calibration/screenSize.mat (MATLAB 5): width_pixel, height_pixel, width_mm, height_mm.
- pXX/Calibration/monitorPose.mat (MATLAB 5): screen-plane pose in camera coordinates. The key
  is `rvects` (the readme says `rvecs`) plus `tvecs`. Verified: R(rvects) [x_mm, y_mm, 0] + tvecs
  reproduces fields 25-27 exactly (max error 0.000 mm, all 15 participants).
- pXX/Calibration/Camera.mat: intrinsics (not used here).

Targets are returned on the screen plane in mm relative to the camera's position on that plane
(-R^T t), so they share an origin across laptops; Euclidean distances equal screen-plane mm.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import scipy.io as sio

PARTICIPANTS = [f"p{i:02d}" for i in range(15)]


@dataclass
class Participant:
    pid: str
    image_paths: list[Path]
    target_px: np.ndarray          # (n, 2) screen px
    screen_px: tuple[float, float] # (width, height)
    screen_mm: tuple[float, float]
    R: np.ndarray                  # (3, 3) screen plane -> camera rotation
    t: np.ndarray                  # (3,) mm
    gt_cam: np.ndarray             # (n, 3) annotated 3D gaze target, camera coords, mm

    @property
    def mm_per_px(self) -> np.ndarray:
        return np.array([self.screen_mm[0] / self.screen_px[0], self.screen_mm[1] / self.screen_px[1]])

    @property
    def camera_on_screen_mm(self) -> np.ndarray:
        return (-self.R.T @ self.t)[:2]

    def target_mm(self) -> np.ndarray:
        """(n, 2) gaze targets on the screen plane, mm, origin = the camera's position on that plane."""
        return px_to_mm(self.target_px, self.mm_per_px) - self.camera_on_screen_mm

    def target_frac(self) -> np.ndarray:
        """(n, 2) targets as fractions of screen width/height (0..1)."""
        return self.target_px / np.array(self.screen_px)


def px_to_mm(px: np.ndarray, mm_per_px: np.ndarray) -> np.ndarray:
    return np.asarray(px, float) * mm_per_px


def mm_to_px(mm: np.ndarray, mm_per_px: np.ndarray) -> np.ndarray:
    return np.asarray(mm, float) / mm_per_px


def parse_annotation(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(image paths, target px (n, 2), gaze target camera mm (n, 3)) from a pXX.txt file."""
    names, tgt, gt = [], [], []
    for line in Path(path).read_text().splitlines():
        f = line.split()
        if not f:
            continue
        if len(f) != 28:
            raise ValueError(f"{path}: expected 28 fields, got {len(f)}: {line[:60]}")
        names.append(f[0])
        tgt.append((float(f[1]), float(f[2])))
        gt.append((float(f[24]), float(f[25]), float(f[26])))
    return names, np.array(tgt), np.array(gt)


def _mat(path: Path) -> dict:
    return {k: np.squeeze(v) for k, v in sio.loadmat(str(path)).items() if not k.startswith("__")}


def load_participant(root: str | Path, pid: str) -> Participant:
    root = Path(root) / pid
    names, tgt, gt = parse_annotation(root / f"{pid}.txt")
    ss = _mat(root / "Calibration" / "screenSize.mat")
    mp = _mat(root / "Calibration" / "monitorPose.mat")
    rv = mp["rvects"] if "rvects" in mp else mp["rvecs"]
    R, _ = cv2.Rodrigues(np.asarray(rv, float).reshape(3))
    return Participant(pid, [root / n for n in names], tgt,
                       (float(ss["width_pixel"]), float(ss["height_pixel"])),
                       (float(ss["width_mm"]), float(ss["height_mm"])), R, np.asarray(mp["tvecs"], float).reshape(3), gt)
