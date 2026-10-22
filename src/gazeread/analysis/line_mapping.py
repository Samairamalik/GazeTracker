"""Credit gaze time to text lines and paragraphs (spec 6.4). px and seconds throughout.

Points are (x, y, seconds): fixation centroids with their durations, or individual gaze
samples with the time until the next sample ("samples" mode).

- hard: all of a point's time goes to the nearest line (by distance to line centre), if that
  centre is within `tol_px`.
- soft (default): time is split across the page's lines with Gaussian weights in y,
  sigma = the calibration's vertical RMS error (rms_y).
Points more than `margin_x` left/right of the text column, or more than `margin_y` above the
first / below the last line, are not credited to any line.
"""
from collections.abc import Iterable

import numpy as np

from gazeread.types import LineBox


def _centres(lines: list[LineBox]) -> np.ndarray:
    return np.array([(ln.y0 + ln.y1) / 2 for ln in lines])


def hard_weights(y: float, lines: list[LineBox], tol_px: float) -> np.ndarray:
    d = np.abs(_centres(lines) - y)
    w = np.zeros(len(lines))
    k = int(np.argmin(d))
    if d[k] <= tol_px:
        w[k] = 1.0
    return w


def soft_weights(y: float, lines: list[LineBox], sigma_y: float) -> np.ndarray:
    z2 = ((_centres(lines) - y) / sigma_y) ** 2
    w = np.exp(-0.5 * (z2 - z2.min()))          # shift so the nearest line has weight 1 (no underflow)
    return w / w.sum()


def in_text_area(x: float, y: float, lines: list[LineBox], margin_x: float, margin_y: float) -> bool:
    x0 = min(ln.x0 for ln in lines) - margin_x
    x1 = max(ln.x1 for ln in lines) + margin_x
    y0 = min(ln.y0 for ln in lines) - margin_y
    y1 = max(ln.y1 for ln in lines) + margin_y
    return x0 <= x <= x1 and y0 <= y <= y1


def line_dwell(points: Iterable[tuple[float, float, float]], lines: list[LineBox], mode: str,
               sigma_y: float, tol_px: float, margin_x: float, margin_y: float) -> np.ndarray:
    """Seconds credited to each line of one page, shape (len(lines),)."""
    out = np.zeros(len(lines))
    for x, y, s in points:
        if s <= 0 or not in_text_area(x, y, lines, margin_x, margin_y):
            continue
        w = soft_weights(y, lines, sigma_y) if mode == "soft" else hard_weights(y, lines, tol_px)
        out += s * w
    return out


def para_dwell(line_s: np.ndarray, lines: list[LineBox]) -> tuple[list[int], np.ndarray]:
    """Sum line seconds per paragraph, in order of first appearance."""
    ids = list(dict.fromkeys(ln.para_id for ln in lines))
    return ids, np.array([sum(s for s, ln in zip(line_s, lines) if ln.para_id == p) for p in ids])


def normalize(v: np.ndarray) -> np.ndarray:
    """Fractions of the page total (all zeros if nothing was credited)."""
    total = v.sum()
    return v / total if total > 0 else np.zeros_like(v)


def sample_points(t, x, y, valid, max_dt_s: float) -> list[tuple[float, float, float]]:
    """Each valid sample credited with the time to the next sample, capped at max_dt_s."""
    t, x, y = (np.asarray(a, float) for a in (t, x, y))
    ok = np.asarray(valid, bool) & np.isfinite(x) & np.isfinite(y)
    dt = np.minimum(np.diff(t, append=t[-1] if len(t) else 0.0), max_dt_s)
    return [(float(x[i]), float(y[i]), float(dt[i])) for i in range(len(t)) if ok[i]]
