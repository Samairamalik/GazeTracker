import math

import pytest

from gazeread.gaze.filters import MovingAverage2D, OneEuroFilter2D

FILTERS = [lambda: OneEuroFilter2D(1.0, 0.005, 1.0, max_hold=0.15), lambda: MovingAverage2D(5, max_hold=0.15)]


@pytest.mark.parametrize("make", FILTERS)
def test_constant_input_constant_output(make):
    f = make()
    for k in range(30):
        p = f(k / 30, 500.0, 300.0, True)
    assert p.valid and math.isclose(p.x, 500.0) and math.isclose(p.y, 300.0)


@pytest.mark.parametrize("make", FILTERS)
def test_short_gap_holds_last_value(make):
    f = make()
    for k in range(10):
        f(k / 30, 100.0, 200.0, True)
    p = f(9 / 30 + 0.10, math.nan, math.nan, False)
    assert p.valid and math.isclose(p.x, 100.0) and math.isclose(p.y, 200.0)


@pytest.mark.parametrize("make", FILTERS)
def test_long_gap_marks_invalid_and_resets(make):
    f = make()
    for k in range(10):
        f(k / 30, 100.0, 200.0, True)
    p = f(9 / 30 + 0.20, math.nan, math.nan, False)
    assert not p.valid and math.isnan(p.x)
    p = f(1.0, 800.0, 50.0, True)                 # after reset: no smoothing toward the old point
    assert p.valid and math.isclose(p.x, 800.0) and math.isclose(p.y, 50.0)


def test_one_euro_smooths_jitter():
    f = OneEuroFilter2D(1.0, 0.005, 1.0)
    out = [f(k / 30, 500.0 + (20 if k % 2 else -20), 300.0, True).x for k in range(60)]
    assert max(abs(x - 500.0) for x in out[30:]) < 10
