import numpy as np

from gazeread.analysis.fixations import idt, merge
from gazeread.types import Fixation

FPS = 30.0


def trace(segments, noise=0.0, seed=0):
    """segments: list of (x, y, seconds). Samples at 30 fps; returns t, x, y, valid."""
    rng = np.random.default_rng(seed)
    t, x, y = [], [], []
    t0 = 0.0
    for sx, sy, dur in segments:
        n = int(round(dur * FPS))
        t += list(t0 + np.arange(n) / FPS)
        x += list(sx + rng.normal(0, noise, n))
        y += list(sy + rng.normal(0, noise, n))
        t0 += n / FPS
    return np.array(t), np.array(x), np.array(y), np.ones(len(t), bool)


def test_three_fixations_with_saccades():
    frames = (9, 8, 12)                            # whole frames at 30 fps
    t, x, y, v = trace([(100, 100, frames[0] / FPS), (400, 100, frames[1] / FPS), (700, 300, frames[2] / FPS)])
    fx = idt(t, x, y, v, dispersion_px=50, min_dur_s=0.1)
    assert len(fx) == 3
    np.testing.assert_allclose([(f.x, f.y) for f in fx], [(100, 100), (400, 100), (700, 300)])
    # duration = first to last sample time, i.e. (frames - 1) sample intervals
    np.testing.assert_allclose([f.dur for f in fx], [(n - 1) / FPS for n in frames], atol=1e-9)


def test_noise_below_threshold_stays_one_fixation():
    t, x, y, v = trace([(500, 400, 1.0)], noise=5, seed=1)
    fx = idt(t, x, y, v, dispersion_px=60, min_dur_s=0.1)
    assert len(fx) == 1 and abs(fx[0].x - 500) < 5 and abs(fx[0].y - 400) < 5


def test_too_short_is_not_a_fixation():
    t, x, y, v = trace([(100, 100, 0.3), (600, 100, 2 / FPS), (900, 100, 0.3)])
    fx = idt(t, x, y, v, dispersion_px=50, min_dur_s=0.1)
    assert [round(f.x) for f in fx] == [100, 900]


def test_invalid_gap_splits_fixation():
    t, x, y, v = trace([(300, 300, 1.0)])
    v[14:17] = False                              # 100 ms blink in the middle
    fx = idt(t, x, y, v, dispersion_px=50, min_dur_s=0.1)
    assert len(fx) == 2
    assert fx[0].t1 < t[14] and fx[1].t0 > t[16]


def test_nan_samples_are_treated_as_invalid():
    t, x, y, v = trace([(300, 300, 0.5)])
    x[5] = np.nan
    fx = idt(t, x, y, v, dispersion_px=50, min_dur_s=0.1)
    assert len(fx) == 2


def test_merge_short_gap_nearby():
    a, b = Fixation(0.0, 0.3, 100, 100), Fixation(0.35, 0.65, 110, 100)
    m = merge([a, b], max_gap_s=0.075, max_dist_px=50)
    assert len(m) == 1 and m[0].t0 == 0.0 and m[0].t1 == 0.65 and np.isclose(m[0].x, 105)


def test_merge_keeps_far_or_late_fixations():
    a = Fixation(0.0, 0.3, 100, 100)
    assert len(merge([a, Fixation(0.35, 0.6, 400, 100)], 0.075, 50)) == 2   # too far
    assert len(merge([a, Fixation(0.5, 0.8, 105, 100)], 0.075, 50)) == 2    # gap too long
