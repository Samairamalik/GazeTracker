"""Gaze smoothing on (x, y) px (spec 6.2).

Both filters handle invalid frames the same way: hold the last output for at most
`max_hold` seconds, then emit invalid points (a gap) and reset.
"""
import math
from collections import deque

from gazeread.types import GazePoint


def _alpha(cutoff_hz: float, dt: float) -> float:
    tau = 1.0 / (2 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class _GapHandler:
    def __init__(self, max_hold: float):
        self.max_hold = max_hold
        self.last: tuple[float, float] | None = None
        self.t_last_valid = -math.inf

    def invalid(self, t: float) -> GazePoint:
        if self.last is not None and t - self.t_last_valid <= self.max_hold:
            return GazePoint(t, *self.last, True)
        self.reset()
        return GazePoint(t, math.nan, math.nan, False)


class OneEuroFilter2D(_GapHandler):
    """One Euro filter (Casiez et al. 2012) per axis. Cutoffs in Hz, beta in 1/px, time in s."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.005, d_cutoff: float = 1.0,
                 max_hold: float = 0.15):
        super().__init__(max_hold)
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.reset()

    def reset(self) -> None:
        self.last, self.t_last_valid = None, -math.inf
        self._dx = [0.0, 0.0]

    def __call__(self, t: float, x: float, y: float, valid: bool) -> GazePoint:
        if not valid:
            return self.invalid(t)
        if self.last is None:
            self.last, self.t_last_valid = (x, y), t
            return GazePoint(t, x, y, True)
        dt = max(t - self.t_last_valid, 1e-3)
        out = []
        for k, v in enumerate((x, y)):
            prev = self.last[k]
            a_d = _alpha(self.d_cutoff, dt)
            self._dx[k] = a_d * (v - prev) / dt + (1 - a_d) * self._dx[k]
            a = _alpha(self.min_cutoff + self.beta * abs(self._dx[k]), dt)
            out.append(a * v + (1 - a) * prev)
        self.last, self.t_last_valid = (out[0], out[1]), t
        return GazePoint(t, out[0], out[1], True)


class MovingAverage2D(_GapHandler):
    """Mean of the last `n` valid samples (ablation baseline)."""

    def __init__(self, n: int = 5, max_hold: float = 0.15):
        super().__init__(max_hold)
        self.n = n
        self.reset()

    def reset(self) -> None:
        self.last, self.t_last_valid = None, -math.inf
        self._buf = deque(maxlen=self.n)

    def __call__(self, t: float, x: float, y: float, valid: bool) -> GazePoint:
        if not valid:
            return self.invalid(t)
        self._buf.append((x, y))
        mx = sum(p[0] for p in self._buf) / len(self._buf)
        my = sum(p[1] for p in self._buf) / len(self._buf)
        self.last, self.t_last_valid = (mx, my), t
        return GazePoint(t, mx, my, True)


def make_filter(cfg: dict):
    """Build the filter named by cfg['kind'] ('one_euro' or 'moving_average')."""
    if cfg["kind"] == "one_euro":
        return OneEuroFilter2D(cfg["min_cutoff_hz"], cfg["beta"], cfg["d_cutoff_hz"], cfg["max_hold_s"])
    if cfg["kind"] == "moving_average":
        return MovingAverage2D(cfg["ma_n"], cfg["max_hold_s"])
    raise ValueError(f"unknown filter kind {cfg['kind']!r}")
