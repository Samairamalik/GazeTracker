import numpy as np
import pytest

from gazeread.gaze.calibration import (
    RidgeMap, Scaler, aggregate_dot, grid_targets, loo_residuals, select_model, validation_error,
    validation_targets,
)


def test_known_linear_map_recovered_exactly():
    rng = np.random.default_rng(0)
    B, c = rng.normal(size=(6, 2)) * 500, np.array([960.0, 540.0])
    X = rng.normal(size=(16, 6))
    m = RidgeMap(1, 0.0, Scaler.fit(X)).fit(X, X @ B + c)
    Xn = rng.normal(size=(50, 6))
    np.testing.assert_allclose(m.predict(Xn), Xn @ B + c, atol=1e-6)
    np.testing.assert_allclose(m.predict(Xn[0]), Xn[0] @ B + c, atol=1e-6)


@pytest.mark.parametrize("degree,alpha", [(1, 0.0), (1, 0.3), (2, 0.01), (2, 10.0)])
def test_closed_form_loo_matches_brute_force(degree, alpha):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(16, 4))
    Y = rng.normal(size=(16, 2)) * 100
    scaler = Scaler.fit(X)
    fast = loo_residuals(X, Y, degree, alpha, scaler)
    brute = np.array([
        Y[i] - RidgeMap(degree, alpha, scaler).fit(np.delete(X, i, 0), np.delete(Y, i, 0)).predict(X[i])
        for i in range(len(X))
    ])
    np.testing.assert_allclose(fast, brute, rtol=1e-7, atol=1e-7)


def test_selection_prefers_affine_on_linear_data():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(9, 4))
    Y = X @ rng.normal(size=(4, 2)) * 300 + 500
    model, info = select_model(X, Y)
    assert info["degree"] == 1
    assert info["loo_rms_px"] < 0.5
    np.testing.assert_allclose(model.predict(X), Y, atol=0.5)


def test_selection_picks_poly2_on_quadratic_data():
    rng = np.random.default_rng(3)
    X = rng.uniform(-1, 1, size=(16, 2))
    Y = np.column_stack([400 * X[:, 0] ** 2 + 100 * X[:, 1], 300 * X[:, 0] * X[:, 1]])
    _, info = select_model(X, Y)
    assert info["degree"] == 2


@pytest.mark.parametrize("n", [9, 16])
def test_grid_targets_count_and_inset(n):
    T = grid_targets(n, 1000, 500, inset=0.08)
    assert T.shape == (n, 2)
    assert np.isclose(T[:, 0].min(), 80) and np.isclose(T[:, 0].max(), 920)
    assert np.isclose(T[:, 1].min(), 40) and np.isclose(T[:, 1].max(), 460)


def test_grid_targets_rejects_non_square():
    with pytest.raises(ValueError):
        grid_targets(10, 100, 100)


def test_validation_targets_not_on_grid():
    V = validation_targets(1000, 500)
    for n in (9, 16):
        G = grid_targets(n, 1000, 500)
        assert np.min(np.linalg.norm(V[:, None] - G[None], axis=2)) > 20


def test_aggregate_dot_window_and_validity():
    ts = np.arange(0, 1.5, 1 / 30)
    vecs = np.zeros((len(ts), 2))
    vecs[ts < 0.4] = 99.0                       # saccade period: must be ignored
    vecs[ts > 1.4] = -99.0                      # last 100 ms: must be ignored
    vecs[(ts >= 0.4) & (ts <= 1.4), 0] = 5.0
    valid = np.ones(len(ts), bool)
    valid[20] = False
    vecs[20] = 1e6                              # invalid sample: must be ignored
    out = aggregate_dot(ts, vecs, valid, 0.0, 1.5)
    np.testing.assert_allclose(out, [5.0, 0.0])


def test_aggregate_dot_too_few_samples():
    ts = np.arange(0, 1.5, 1 / 30)
    valid = np.zeros(len(ts), bool)
    valid[15:18] = True
    assert aggregate_dot(ts, np.zeros((len(ts), 2)), valid, 0.0, 1.5, min_samples=5) is None


def test_validation_error():
    err = validation_error(np.array([[3.0, 4.0], [0.0, 0.0]]), np.zeros((2, 2)))
    assert np.isclose(err["sigma_px"], np.sqrt(12.5))
    assert np.isclose(err["rms_x"], np.sqrt(4.5)) and np.isclose(err["rms_y"], np.sqrt(8.0))


def test_shuffled_is_seeded_permutation_not_row_major():
    from gazeread.gaze.calibration import shuffled
    G = grid_targets(9, 1000, 500)
    S = shuffled(G, seed=7)
    assert sorted(map(tuple, S)) == sorted(map(tuple, G))
    np.testing.assert_array_equal(S, shuffled(G, seed=7))
    assert not np.array_equal(S, G)
