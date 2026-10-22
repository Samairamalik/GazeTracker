import numpy as np
import yaml

from gazeread.sim.scanpath import make_layout, sample_gaze, simulate_reading, true_dwell

CFG = yaml.safe_load(open("configs/sim.yaml"))


def run(seed=0, pitch=66.0, para_lines=None):
    rng = np.random.default_rng(seed)
    lay = make_layout(pitch, CFG, rng, para_lines)
    return lay, simulate_reading(lay, CFG, rng)


def test_deterministic_per_seed():
    _, a = run(3)
    _, b = run(3)
    np.testing.assert_array_equal(a.fix_x, b.fix_x)
    np.testing.assert_array_equal(a.fix_t0, b.fix_t0)


def test_fixations_inside_their_line_and_in_time_order():
    lay, rd = run(1)
    lines = {ln.line_id: ln for pg in lay.pages for ln in pg}
    for x, y, lid in zip(rd.fix_x, rd.fix_y, rd.fix_line):
        ln = lines[lid]
        assert ln.x0 <= x <= ln.x1 + 1e-9
        assert ln.y0 - 0.25 * lay.pitch <= y <= ln.y1 + 0.25 * lay.pitch
    assert np.all(np.diff(rd.fix_t0) > 0) and np.all(rd.fix_t1 > rd.fix_t0)


def test_true_dwell_sums_to_fixation_time():
    lay, rd = run(2)
    n = sum(len(pg) for pg in lay.pages)
    assert np.isclose(true_dwell(rd, n).sum(), (rd.fix_t1 - rd.fix_t0).sum())


def test_reading_statistics_near_parameters():
    durs, sk = [], []
    for s in range(40):
        lay, rd = run(s)
        durs += list(rd.fix_t1 - rd.fix_t0)
        n = sum(len(pg) for pg in lay.pages)
        d = true_dwell(rd, n)
        paras = np.array([ln.para_id for pg in lay.pages for ln in pg])
        sk.append(np.mean([d[paras == p].sum() == 0 or d[paras == p].min() == 0 for p in np.unique(paras)]))
    assert 0.17 < np.mean(durs) < 0.24          # mix of 225 ms reading and 150 ms skim fixations
    assert 0.05 < np.mean(sk) < 0.45            # some paragraphs skimmed (lines skipped)


def test_layout_feasibility_and_paragraphs_whole():
    lay = make_layout(66, CFG, np.random.default_rng(0), [10, 10])
    assert lay.feasible and len(lay.pages) == 2
    assert not make_layout(200, CFG, np.random.default_rng(0), [6]).feasible       # 6 x 200 > 825 px
    big = make_layout(200, CFG, np.random.default_rng(0), [4])
    assert big.chars_per_line < CFG["layout"]["column_ch"]                          # large font: fewer chars


def test_noise_rms_and_systematic_split():
    lay, rd = run(5)
    ry, rx = [], []
    for s in range(20):
        t, x, y, v, tl = sample_gaze(rd, 100.0, 0.7, 0.8, CFG, np.random.default_rng(100 + s))
        _, x0, y0, _, _ = sample_gaze(rd, 1e-9, 0.7, 0.8, CFG, np.random.default_rng(100 + s))
        ry.append(np.sqrt(np.mean((y - y0)[v] ** 2)))
        rx.append(np.sqrt(np.mean((x - x0)[v] ** 2)))
    assert 70 < np.mean(ry) < 130 and 50 < np.mean(rx) < 90       # sigma_y 100, sigma_x 70 (drift is slow)
    _, _, _, v, tl = sample_gaze(rd, 50.0, 0.7, 0.8, CFG, np.random.default_rng(1))
    assert 0.03 < 1 - v.mean() < 0.12                             # blinks + page turns
    assert (tl >= 0).mean() > 0.7                                 # most samples during fixations
