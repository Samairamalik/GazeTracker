"""I-DT fixation detection (spec 6.3). Coordinates in viewport CSS px, times in seconds.

At ~30 fps a 100 ms fixation spans only 3-4 samples, and with webcam noise the dispersion
threshold (~1.25 x sigma_px) is often larger than a reading saccade, so consecutive words
tend to merge into one long fixation. Report this limitation with any result.
"""
import numpy as np

from gazeread.types import Fixation


def _dispersion(x: np.ndarray, y: np.ndarray) -> float:
    return float(x.max() - x.min() + y.max() - y.min())


def idt(t, x, y, valid, dispersion_px: float, min_dur_s: float) -> list[Fixation]:
    """Dispersion-threshold fixations over runs of consecutive valid samples.

    A window must span at least `min_dur_s` (first to last sample time) with dispersion
    (x range + y range) <= `dispersion_px`; it then grows while the dispersion stays under
    the threshold. Invalid or non-finite samples always end a window.
    """
    t, x, y = (np.asarray(a, float) for a in (t, x, y))
    ok = np.asarray(valid, bool) & np.isfinite(x) & np.isfinite(y)
    n, i, out = len(t), 0, []
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and ok[j + 1] and t[j] - t[i] < min_dur_s:
            j += 1
        if t[j] - t[i] < min_dur_s:
            i = j + 1                      # run ended (gap or end of data) before min duration
            continue
        if _dispersion(x[i:j + 1], y[i:j + 1]) > dispersion_px:
            i += 1
            continue
        while j + 1 < n and ok[j + 1] and _dispersion(x[i:j + 2], y[i:j + 2]) <= dispersion_px:
            j += 1
        out.append(Fixation(float(t[i]), float(t[j]), float(x[i:j + 1].mean()), float(y[i:j + 1].mean())))
        i = j + 1
    return out


def merge(fixations: list[Fixation], max_gap_s: float, max_dist_px: float) -> list[Fixation]:
    """Join consecutive fixations separated by < max_gap_s and with centroids < max_dist_px apart.
    The merged centroid is duration-weighted."""
    out: list[Fixation] = []
    for f in fixations:
        if out:
            p = out[-1]
            if f.t0 - p.t1 < max_gap_s and np.hypot(f.x - p.x, f.y - p.y) < max_dist_px:
                wp, wf = max(p.dur, 1e-9), max(f.dur, 1e-9)
                out[-1] = Fixation(p.t0, f.t1, (p.x * wp + f.x * wf) / (wp + wf),
                                   (p.y * wp + f.y * wf) / (wp + wf))
                continue
        out.append(f)
    return out
