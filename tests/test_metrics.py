import numpy as np

from gazeread.eval.metrics import assignment_accuracy, bootstrap_ci, rankdata, spearman
from gazeread.eval.sim_experiments import threshold, within_para_rho
from gazeread.types import LineBox


def test_rankdata_ties():
    np.testing.assert_allclose(rankdata([10, 20, 20, 30]), [1, 2.5, 2.5, 4])


def test_spearman_known_values():
    assert np.isclose(spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
    assert np.isclose(spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)
    assert np.isclose(spearman([1, 2, 3, 4, 5], [2, 1, 4, 3, 5]), 0.8)
    assert np.isnan(spearman([1, 1, 1], [1, 2, 3]))


def test_shuffled_estimate_near_zero():
    rng = np.random.default_rng(0)
    vals = [spearman(rng.random(30), rng.random(30)) for _ in range(300)]
    assert abs(np.mean(vals)) < 0.05


def test_within_para_rho_ignores_between_paragraph_contrast():
    truth = np.array([1, 2, 3, 10, 11, 12.0])
    paras = np.array([0, 0, 0, 1, 1, 1])
    est = np.array([3, 2, 1, 12, 11, 10.0])      # paragraph level right, line order reversed
    assert np.isclose(spearman(truth, est), 0.5429, atol=1e-3)   # pooled rho looks positive...
    assert np.isclose(within_para_rho(truth, est, paras), -1.0)   # ...within paragraphs it is wrong


def lines():
    return [LineBox(0, i, i // 2, 100, 100 + 66 * i, 900, 166 + 66 * i, "x") for i in range(4)]


def test_perfect_assignment_accuracy():
    pts = [(500, 133 + 66 * i, 0.5, i) for i in range(4)]
    a, b, c = assignment_accuracy(pts, lines(), "hard", 20, 66, 50, 50)
    assert np.isclose(a / c, 1.0) and np.isclose(b / c, 1.0)


def test_wrong_line_right_paragraph():
    a, b, c = assignment_accuracy([(500, 133 + 66, 1.0, 0)], lines(), "hard", 20, 66, 50, 50)
    assert a == 0 and np.isclose(b / c, 1.0)


def test_bootstrap_ci_brackets_mean():
    mean, lo, hi = bootstrap_ci([0.2, 0.4, 0.6, 0.8, float("nan")])
    assert np.isclose(mean, 0.5) and lo < mean < hi


def test_threshold_interpolation_and_feasibility():
    assert threshold([50, 100, 150], [0.5, 0.6, 0.8], 0.7, [1, 1, 1]) == "125"
    assert threshold([50, 100, 150], [0.5, 0.6, 0.8], 0.7, [1, 1, 0]).startswith("not reached")
    assert threshold([50, 100], [0.9, 0.95], 0.7, [1, 1]) == "<= 50"


def test_threshold_whole_rounds_up():
    assert threshold([4, 6], [0.64, 0.96], 0.7, [1, 1], whole=True) == "5"
