import numpy as np

from gazeread.analysis.line_mapping import (
    hard_weights, line_dwell, normalize, para_dwell, sample_points, soft_weights,
)
from gazeread.types import LineBox

PITCH = 66.0


def layout(n=6, paras=(0, 0, 0, 1, 1, 1)):
    """n lines, 66 px pitch from y=100, text column x 200..1000."""
    return [LineBox(page=0, line_id=i, para_id=paras[i], x0=200, y0=100 + i * PITCH,
                    x1=1000, y1=100 + (i + 1) * PITCH, text=f"line {i}") for i in range(n)]


def centre(i):
    return 100 + (i + 0.5) * PITCH


def test_hard_picks_line_at_centre():
    L = layout()
    for i in range(len(L)):
        assert np.argmax(hard_weights(centre(i), L, tol_px=PITCH)) == i


def test_hard_outside_tolerance_gets_nothing():
    assert hard_weights(centre(5) + 200, layout(), tol_px=PITCH).sum() == 0


def test_soft_weights_sum_to_one_and_peak_at_nearest():
    w = soft_weights(centre(2) + 10, layout(), sigma_y=40)
    assert np.isclose(w.sum(), 1.0) and np.argmax(w) == 2


def test_soft_large_sigma_spreads_evenly():
    w = soft_weights(centre(2.5), layout(), sigma_y=1e5)
    np.testing.assert_allclose(w, np.full(6, 1 / 6), atol=1e-6)


def test_soft_is_stable_far_away():
    w = soft_weights(1e6, layout(), sigma_y=5)          # would underflow without the shift
    assert np.isclose(w.sum(), 1.0) and np.argmax(w) == 5


def test_dwell_conserves_time_and_paragraph_sums():
    L = layout()
    pts = [(500, centre(1), 0.4), (600, centre(4), 0.6)]
    d = line_dwell(pts, L, "soft", sigma_y=30, tol_px=PITCH, margin_x=50, margin_y=60)
    assert np.isclose(d.sum(), 1.0)
    ids, ps = para_dwell(d, L)
    assert ids == [0, 1] and np.isclose(ps.sum(), 1.0)
    assert ps[0] > 0.35 and ps[1] > 0.55


def test_off_column_points_are_dropped():
    L = layout()
    pts = [(50, centre(1), 1.0), (1200, centre(1), 1.0), (500, 100 - 200, 1.0), (500, centre(1), 0.5)]
    d = line_dwell(pts, L, "hard", sigma_y=30, tol_px=PITCH, margin_x=50, margin_y=60)
    assert np.isclose(d.sum(), 0.5) and np.isclose(d[1], 0.5)


def test_normalize_per_page():
    np.testing.assert_allclose(normalize(np.array([1.0, 3.0])), [0.25, 0.75])
    np.testing.assert_allclose(normalize(np.zeros(3)), np.zeros(3))


def test_sample_points_time_credit():
    t = np.array([0.0, 1 / 30, 2 / 30, 2 / 30 + 0.5])
    pts = sample_points(t, [1, 2, 3, 4], [1, 2, 3, 4], [True, False, True, True], max_dt_s=0.1)
    assert [p[0] for p in pts] == [1, 3, 4]
    np.testing.assert_allclose([p[2] for p in pts], [1 / 30, 0.1, 0.0])
