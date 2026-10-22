"""Evaluation metrics: Spearman correlation, time-weighted assignment accuracy, bootstrap CI."""
import numpy as np

from gazeread.analysis.line_mapping import hard_weights, in_text_area, soft_weights
from gazeread.types import LineBox


def rankdata(a) -> np.ndarray:
    """Ranks starting at 1, ties get their average rank."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a))
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a, b) -> float:
    """Spearman rank correlation; NaN if either input is constant."""
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def assignment_accuracy(points, lines: list[LineBox], mode: str, sigma_y: float, tol_px: float,
                        margin_x: float, margin_y: float) -> tuple[float, float, float]:
    """Time-weighted share of credited time that lands on the true line / true paragraph.

    points: (x, y, seconds, true_line_id) with true_line_id >= 0. Returns
    (line_correct_s, para_correct_s, credited_s) so callers can pool over pages."""
    ids = np.array([ln.line_id for ln in lines])
    paras = np.array([ln.para_id for ln in lines])
    line_ok = para_ok = credited = 0.0
    for x, y, s, true_id in points:
        if s <= 0 or not in_text_area(x, y, lines, margin_x, margin_y):
            continue
        w = soft_weights(y, lines, sigma_y) if mode == "soft" else hard_weights(y, lines, tol_px)
        if w.sum() == 0:
            continue
        credited += s * w.sum()
        line_ok += s * w[ids == true_id].sum()
        true_para = paras[ids == true_id]
        if len(true_para):
            para_ok += s * w[paras == true_para[0]].sum()
    return line_ok, para_ok, credited


def bootstrap_ci(values, n_boot: int = 1000, seed: int = 0, level: float = 0.95) -> tuple[float, float, float]:
    """(mean, lo, hi) of the mean over `values` (NaNs dropped)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    a = (1 - level) / 2
    return float(v.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))
