import json

import numpy as np
import pytest
import yaml

from gazeread.analysis.heatmap import analyze_session, choose_granularity, dumps, export
from gazeread.app.session import SessionLog, read_events

CFG = yaml.safe_load(open("configs/default.yaml"))["analysis"]
PITCH = 66.0


def lines(page, first_id, paras):
    return [{"line_id": first_id + i, "para_id": p, "x0": 200.0, "y0": 100 + i * PITCH, "x1": 1000.0,
             "y1": 100 + (i + 1) * PITCH, "text": f"line {first_id + i}"} for i, p in enumerate(paras)]


def make_log(tmp_path, rms_y=20.0):
    """Hand-made session: calibration, 2 pages; gaze dwells 1 s on page-0 line 1, then 2 s on page-1 line 3."""
    s = SessionLog(tmp_path, session_id="s")
    s.write("meta", screen_w=1200, screen_h=800, dpr=2, source="synthetic")
    s.write("calib_result", ok=True, t=0.0, sigma_px=30.0, rms_x=20.0, rms_y=rms_y)
    s.write("page", page=0, t_on=1.0, lines=lines(0, 0, [0, 0, 1, 1]), line_pitch=PITCH, font_px=30, n_pages=2)
    s.write("page", page=1, t_on=2.5, lines=lines(1, 4, [2, 2, 2, 3]), line_pitch=PITCH, font_px=30, n_pages=2)
    s.write("reading_end", t=5.0)
    rng = np.random.default_rng(0)
    for k in range(int(4.5 * 30)):
        t = 0.6 + k / 30
        yline = 1 if t < 2.5 else 3
        y = 100 + (yline + 0.5) * PITCH + rng.normal(0, 3)
        s.write("gaze", t=t, x=500 + rng.normal(0, 3), y=y, valid=bool(k % 20))
    s.close()
    return read_events(tmp_path / "s" / "events.jsonl")


def test_granularity_rule():
    assert choose_granularity(30, 66) == "line"
    assert choose_granularity(34, 66) == "paragraph"


def test_analysis_finds_the_attended_lines(tmp_path):
    r = analyze_session(make_log(tmp_path), CFG)
    assert r["granularity"] == "line" and len(r["pages"]) == 2
    for page, line in ((0, 1), (1, 3)):
        for src in ("fixations", "samples"):
            frac = r["pages"][page]["dwell"][src]["line_frac"]
            assert int(np.argmax(frac)) == line, (page, src, frac)
            assert np.isclose(sum(frac), 1.0, atol=1e-3)
    assert r["pages"][0]["fixations"], "expected at least one fixation"


def test_paragraph_granularity_when_vertical_error_large(tmp_path):
    assert analyze_session(make_log(tmp_path, rms_y=120.0), CFG)["granularity"] == "paragraph"


def test_replay_is_deterministic_and_json_stable(tmp_path):
    ev = make_log(tmp_path)
    a, b = dumps(analyze_session(ev, CFG)), dumps(analyze_session(ev, CFG))
    assert a == b
    assert dumps(json.loads(a)) == a


def test_incomplete_reading_raises(tmp_path):
    ev = [e for e in make_log(tmp_path) if e["type"] != "reading_end"]
    with pytest.raises(ValueError):
        analyze_session(ev, CFG)


def test_export_writes_json_and_pngs(tmp_path):
    r = analyze_session(make_log(tmp_path), CFG)
    paths = export(r, tmp_path / "out")
    assert [p.name for p in paths] == ["heatmap.json", "heatmap_p1.png", "scanpath_p1.png",
                                       "heatmap_p2.png", "scanpath_p2.png"]
    assert all(p.stat().st_size > 1000 for p in paths[1:])
