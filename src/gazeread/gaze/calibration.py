"""Per-user calibration map: raw features -> viewport CSS px (spec 6.1).

Ridge regression with an unpenalized intercept on standardized features, either affine
(degree 1) or full quadratic (degree 2). Model and ridge strength are chosen by exact
leave-one-dot-out (LOO) error computed in closed form from the hat matrix. The feature
standardization is fixed from all calibration dots (it does not use the targets).
"""
from dataclasses import dataclass, field

import numpy as np

# Validation dots as viewport fractions; none coincide with a 3x3 or 4x4 grid point.
VALIDATION_FRACS = ((0.3, 0.3), (0.7, 0.3), (0.5, 0.55), (0.3, 0.75), (0.7, 0.75))


def grid_targets(n: int, w: float, h: float, inset: float = 0.08) -> np.ndarray:
    """n = k*k targets (k x k grid, row-major), inset from the edges by `inset` of w/h. Returns (n, 2) px."""
    k = int(round(np.sqrt(n)))
    if k * k != n:
        raise ValueError(f"n_points must be a square, got {n}")
    xs = np.linspace(inset * w, (1 - inset) * w, k)
    ys = np.linspace(inset * h, (1 - inset) * h, k)
    return np.array([(x, y) for y in ys for x in xs])


def shuffled(targets: np.ndarray, seed: int) -> np.ndarray:
    """Targets in a seeded random order, so slow head drift over time is not aliased with
    screen position (a row-major order would make drift look like vertical gaze)."""
    return targets[np.random.default_rng(seed).permutation(len(targets))]


def validation_targets(w: float, h: float) -> np.ndarray:
    """The 5 held-out validation dots in px, (5, 2)."""
    return np.array([(fx * w, fy * h) for fx, fy in VALIDATION_FRACS])


def window_mask(ts: np.ndarray, t_on: float, t_off: float,
                skip_start: float = 0.4, skip_end: float = 0.1) -> np.ndarray:
    """Samples shown at a dot after the saccade-latency skip and before the end skip (seconds)."""
    return (ts >= t_on + skip_start) & (ts <= t_off - skip_end)


def aggregate_dot(ts: np.ndarray, vecs: np.ndarray, valid: np.ndarray, t_on: float, t_off: float,
                  skip_start: float = 0.4, skip_end: float = 0.1,
                  min_samples: int = 5) -> np.ndarray | None:
    """Median feature vector of the valid samples inside the dot window; None if too few."""
    m = window_mask(np.asarray(ts), t_on, t_off, skip_start, skip_end) & np.asarray(valid, bool)
    if m.sum() < min_samples:
        return None
    return np.median(np.asarray(vecs)[m], axis=0)


@dataclass
class Scaler:
    mu: np.ndarray
    sd: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray) -> "Scaler":
        sd = X.std(axis=0)
        sd[sd < 1e-12] = 1.0
        return cls(X.mean(axis=0), sd)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sd


def expand(Z: np.ndarray, degree: int) -> np.ndarray:
    """Polynomial terms without the constant: degree 1 -> Z; degree 2 -> Z plus all z_i*z_j, i<=j."""
    if degree == 1:
        return Z
    if degree == 2:
        iu, ju = np.triu_indices(Z.shape[1])
        return np.hstack([Z, Z[:, iu] * Z[:, ju]])
    raise ValueError(f"degree must be 1 or 2, got {degree}")


def _design(X: np.ndarray, degree: int, scaler: Scaler) -> np.ndarray:
    Z = expand(scaler(X), degree)
    return np.hstack([np.ones((len(Z), 1)), Z])


def _penalty(p: int, alpha: float) -> np.ndarray:
    P = alpha * np.eye(p)
    P[0, 0] = 0.0                      # intercept is not penalized
    return P


@dataclass
class RidgeMap:
    """Ridge map from features (d,) to screen (x, y) px."""
    degree: int
    alpha: float
    scaler: Scaler
    W: np.ndarray = field(default=None, repr=False)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "RidgeMap":
        A = _design(X, self.degree, self.scaler)
        self.W = np.linalg.solve(A.T @ A + _penalty(A.shape[1], self.alpha), A.T @ Y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """(d,) -> (2,) or (m, d) -> (m, 2), in px."""
        X = np.asarray(X, float)
        out = _design(np.atleast_2d(X), self.degree, self.scaler) @ self.W
        return out[0] if X.ndim == 1 else out


def loo_residuals(X: np.ndarray, Y: np.ndarray, degree: int, alpha: float,
                  scaler: Scaler | None = None) -> np.ndarray:
    """Exact leave-one-out residuals (n, 2) px via r_i / (1 - H_ii). inf where H_ii ~ 1."""
    scaler = scaler or Scaler.fit(X)
    A = _design(X, degree, scaler)
    M = np.linalg.solve(A.T @ A + _penalty(A.shape[1], alpha), A.T)   # (p, n)
    R = Y - A @ (M @ Y)
    h = np.einsum("ij,ji->i", A, M)
    denom = 1.0 - h
    out = np.full_like(R, np.inf)
    ok = denom > 1e-10
    out[ok] = R[ok] / denom[ok, None]
    return out


def rms_px(R: np.ndarray) -> float:
    """RMS Euclidean error (px) of (n, 2) residuals."""
    return float(np.sqrt(np.mean(np.sum(R ** 2, axis=1))))


def select_model(X: np.ndarray, Y: np.ndarray, degrees=(1, 2), alphas=None,
                 poly2_margin: float = 0.10) -> tuple[RidgeMap, dict]:
    """Pick degree and alpha by LOO RMS. Degree 2 wins only if >= poly2_margin better than degree 1."""
    alphas = np.logspace(-4, 3, 15) if alphas is None else np.asarray(alphas)
    scaler = Scaler.fit(X)
    best = {}
    for deg in degrees:
        scores = [rms_px(loo_residuals(X, Y, deg, a, scaler)) for a in alphas]
        i = int(np.argmin(scores))
        best[deg] = (float(alphas[i]), scores[i])
    deg = min(degrees)
    if 2 in best and 1 in best and best[2][1] < (1 - poly2_margin) * best[1][1]:
        deg = 2
    alpha, loo = best[deg]
    model = RidgeMap(deg, alpha, scaler).fit(X, Y)
    info = {"degree": deg, "alpha": alpha, "loo_rms_px": loo,
            "candidates": {str(d): {"alpha": a, "loo_rms_px": s} for d, (a, s) in best.items()}}
    return model, info


def validation_error(pred: np.ndarray, target: np.ndarray) -> dict:
    """Per-frame errors (m, 2) px -> sigma_px (RMS Euclidean), rms_x, rms_y, n."""
    E = np.asarray(pred) - np.asarray(target)
    return {"sigma_px": rms_px(E), "rms_x": float(np.sqrt(np.mean(E[:, 0] ** 2))),
            "rms_y": float(np.sqrt(np.mean(E[:, 1] ** 2))), "n": int(len(E))}
