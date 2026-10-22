import numpy as np
import yaml

from gazeread.eval import lopo
from gazeread.eval.kshot import select_kshot_points


def screen_targets(n_targets=40, reps=20, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.uniform([0, 0], [340, 190], size=(n_targets, 2))
    return np.repeat(t, reps, axis=0)


def test_kshot_spread_across_screen():
    pos = screen_targets()
    for k in (4, 9, 16, 25):
        idx = select_kshot_points(pos, k, seed=1)
        sel = pos[idx]
        assert len(idx) == k == len(np.unique(idx))
        assert len(np.unique(sel, axis=0)) == min(k, 40)                # distinct targets
        if k >= 4:                                                     # covers the screen, not clumped
            assert np.ptp(sel[:, 0]) > 0.5 * 340 and np.ptp(sel[:, 1]) > 0.5 * 190


def test_kshot_deterministic_and_seed_dependent():
    pos = screen_targets()
    np.testing.assert_array_equal(select_kshot_points(pos, 9, 3), select_kshot_points(pos, 9, 3))
    assert not np.array_equal(select_kshot_points(pos, 9, 3), select_kshot_points(pos, 9, 4))


def test_k1_is_central():
    pos = screen_targets()
    p = pos[select_kshot_points(pos, 1, 0)][0]
    c = np.unique(pos, axis=0).mean(0)
    assert np.linalg.norm(p - c) < 0.35 * np.linalg.norm([340, 190])


def test_folds_hold_each_person_out_once():
    people = [f"p{i:02d}" for i in range(15)]
    held = []
    for p, train in lopo.person_folds(people):
        assert p not in train and len(train) == 14
        held.append(p)
    assert sorted(held) == people


def fake_people(n=5, frames=120, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for i in range(n):
        tgt = np.repeat(rng.uniform([0, 0], [300, 180], size=(frames // 6, 2)), 6, axis=0)
        A = rng.normal(size=(2, 12)) * 0.01
        feats = tgt @ A + rng.normal(0, 0.05, size=(frames, 12))
        out[f"p{i:02d}"] = {"landmark": feats, "l2cs": feats[:, :4], "valid": rng.random(frames) > 0.05,
                            "target_mm": tgt, "target_frac": tgt / [300, 180], "screen_mm": np.array([300, 180.0])}
    return out


def test_no_calibration_frame_evaluated_and_global_never_sees_heldout(monkeypatch):
    cfg = yaml.safe_load(open("configs/eval_lopo.yaml"))
    cfg.update(k_values=[0, 1, 4, 9], n_seeds=2)
    cfg["global"].update(alphas_log10=[-2, 2, 3], inner_folds=2)
    people = fake_people()
    seen_train = []
    real_fit = lopo.fit_global

    def spy(Xs, Ys, g):
        seen_train.append(sum(len(x) for x in Xs))
        return real_fit(Xs, Ys, g)

    real_select = lopo.select_kshot_points
    evaluated = []

    def spy_select(Y, k, seed):
        idx = real_select(Y, k, seed)
        evaluated.append((len(Y), idx))
        return idx

    monkeypatch.setattr(lopo, "fit_global", spy)
    monkeypatch.setattr(lopo, "select_kshot_points", spy_select)
    rows, _, _ = lopo.evaluate(people, cfg, {"A": ["landmark", list(range(12))]})
    n_valid = {p: int(people[p]["valid"].sum()) for p in people}
    for (p, _), n in zip(lopo.person_folds(sorted(people)), seen_train):    # global fit size = the other people only
        assert n == sum(v for q, v in n_valid.items() if q != p)
    for r in rows:                                                         # evaluated = valid minus calibration
        assert r["n_eval"] == n_valid[r["person"]] - r["k"]
    assert {r["k"] for r in rows} == {0, 1, 4, 9}


def test_correction_shrinks_to_offset_not_to_mean():
    rng = np.random.default_rng(0)
    G = rng.uniform(-100, 100, size=(4, 2))
    Y = G + [5.0, -3.0]                                   # true correction: pure offset
    G_eval = rng.uniform(-100, 100, size=(50, 2))
    for alphas in ([1e6], [1e-3]):                         # heavy or light regularisation
        out = lopo.corrected(G, Y, G_eval, 4, np.array(alphas))
        np.testing.assert_allclose(out, G_eval + [5.0, -3.0], atol=1e-2)


def test_heldout_person_cannot_influence_their_folds_rule_or_hyperparameters():
    cfg = yaml.safe_load(open("configs/eval_lopo.yaml"))
    cfg.update(k_values=[0, 4, 9], n_seeds=1, rule_seeds=1)
    cfg["global"].update(alphas_log10=[-2, 2, 3], inner_folds=2)
    people = fake_people()
    variant = {"A": ["landmark", list(range(12))]}
    _, _, before = lopo.evaluate(people, cfg, variant)
    corrupt = {k: dict(v) for k, v in people.items()}
    rng = np.random.default_rng(9)
    corrupt["p02"]["target_mm"] = rng.permutation(people["p02"]["target_mm"]) * 3   # garbage labels
    corrupt["p02"]["landmark"] = rng.normal(size=people["p02"]["landmark"].shape)
    _, _, after = lopo.evaluate(corrupt, cfg, variant)
    assert before[("A", "p02")] == after[("A", "p02")]      # p02's own fold: identical choices
    assert any(before[("A", q)] != after[("A", q)] for q in people if q != "p02")   # p02 does train others
