"""Core dataclasses shared by every module (SPEC.md section 4.2).

All screen coordinates are viewport CSS pixels in a fullscreen browser window,
origin at the top-left of the reading viewport. Times are seconds (monotonic).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class FrameFeatures:
    t: float                 # seconds, monotonic, from capture time
    valid: bool              # False on blink / no face / extreme pose
    vec: np.ndarray          # feature vector (Method A) or [yaw, pitch] (Method B)


@dataclass
class GazePoint:
    t: float
    x: float                 # viewport CSS px, origin top-left of the reading viewport
    y: float
    valid: bool


@dataclass
class Fixation:
    t0: float; t1: float
    x: float; y: float       # centroid

    @property
    def dur(self) -> float:
        """Duration in seconds."""
        return self.t1 - self.t0


@dataclass
class LineBox:
    page: int; line_id: int; para_id: int
    x0: float; y0: float; x1: float; y1: float   # viewport CSS px
    text: str
