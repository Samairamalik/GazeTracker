"""k-shot calibration frame selection for simulated per-user calibration (spec 6.1, 7.1)."""
import numpy as np


def _kmeans(X: np.ndarray, k: int, rng: np.random.Generator, iters: int = 50) -> np.ndarray:
    """Lloyd's k-means with k-means++ seeding. Returns (k, d) centres."""
    centres = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d2 = np.min(((X[:, None] - np.array(centres)[None]) ** 2).sum(-1), axis=1)
        centres.append(X[rng.choice(len(X), p=d2 / d2.sum())] if d2.sum() > 0 else X[rng.integers(len(X))])
    C = np.array(centres, float)
    for _ in range(iters):
        lab = np.argmin(((X[:, None] - C[None]) ** 2).sum(-1), axis=1)
        newC = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return C


def select_kshot_points(positions: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Indices of k calibration frames spread across the screen (never random-clumped).

    positions: (n, 2) on-screen gaze targets of one person's frames (any unit). k-means on the
    unique target positions; for each centre, the nearest unused unique target, and one frame
    showing it (chosen by `seed`). k == 1 gives the target nearest the centroid. Sorted indices.
    """
    positions = np.asarray(positions, float)
    rng = np.random.default_rng(seed)
    uniq, inverse = np.unique(positions, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    k = min(k, len(positions))
    if k <= 0:
        return np.array([], int)
    centres = uniq.mean(0, keepdims=True) if k == 1 else _kmeans(uniq, min(k, len(uniq)), rng)
    chosen_u, picks = set(), []
    for c in centres:
        order = np.argsort(((uniq - c) ** 2).sum(1))
        u = next(int(j) for j in order if int(j) not in chosen_u)
        chosen_u.add(u)
        picks.append(int(rng.choice(np.flatnonzero(inverse == u))))
    remaining = np.setdiff1d(np.arange(len(positions)), picks)     # k > unique targets: fill randomly
    if len(picks) < k:
        picks += list(rng.choice(remaining, k - len(picks), replace=False))
    return np.sort(np.array(picks, int))
