from pathlib import Path

import numpy as np
import pytest
import yaml

from gazeread.app.session import SessionLog, read_events
from gazeread.eval.human_check import session_tables

CFG = yaml.safe_load(open("configs/default.yaml"))["analysis"]
TEXTS = Path("src/gazeread/app/static/texts")
PITCH = 66.0





def test_session_join(tmp_path):
    """3 paragraphs on one page; gaze stays on paragraph 0; questions: para 0 right, para 2 wrong."""
    s = SessionLog(tmp_path, session_id="h")
    s.write("meta", screen_w=1200, screen_h=800, dpr=2, source="camera", text="alice_passage1", skim=False)
    s.write("calib_result", ok=True, primary=True, t=0.0, sigma_px=30.0, rms_x=20.0, rms_y=20.0)
    lines = [{"line_id": i, "para_id": i // 2, "x0": 200.0, "y0": 100 + i * PITCH, "x1": 1000.0,
              "y1": 100 + (i + 1) * PITCH, "text": "x"} for i in range(6)]
    s.write("page", page=0, t_on=1.0, lines=lines, line_pitch=PITCH, font_px=30, n_pages=1)
    for k in range(90):
        s.write("gaze", t=1.0 + k / 30, x=500.0, y=100 + 0.5 * PITCH, valid=True)
    s.write("reading_end", t=4.0)
    s.write("answer", t=5.0, passage="alice_passage1", question_id="q0", para_id=0, response=1, correct=True)
    s.write("answer", t=6.0, passage="alice_passage1", question_id="q2", para_id=2, response=0, correct=False)
    s.close()
    passage, rows = session_tables(read_events(tmp_path / "h" / "events.jsonl"), CFG, "h")
    assert passage["n_correct"] == 1 and passage["n_questions"] == 2 and passage["skim"] == 0
    by = {r["para_id"]: r for r in rows}
    assert by[0]["dwell_frac_fix"] > 0.9 and by[2]["dwell_frac_fix"] < 0.05
    assert (by[0]["n_correct"], by[0]["n_questions"]) == (1, 1) and (by[2]["n_correct"], by[2]["n_questions"]) == (0, 1)
    assert np.isclose(sum(r["dwell_frac_samples"] for r in rows), 1.0, atol=0.01)


def test_page_shown_too_briefly_is_flagged(tmp_path):
    s = SessionLog(tmp_path, session_id="k")
    s.write("meta", screen_w=1200, screen_h=800, dpr=2, source="camera", text="alice_passage3", skim=False)
    s.write("calib_result", ok=True, primary=True, t=0.0, sigma_px=30.0, rms_x=20.0, rms_y=20.0)
    mk = lambda pid0: [{"line_id": pid0 * 2 + i, "para_id": pid0 + i // 2, "x0": 200.0, "y0": 100 + i * PITCH,
                        "x1": 1000.0, "y1": 100 + (i + 1) * PITCH, "text": "x"} for i in range(2)]
    s.write("page", page=0, t_on=1.0, lines=mk(0), line_pitch=PITCH, font_px=30, n_pages=2)
    s.write("page", page=1, t_on=1.1, lines=mk(1), line_pitch=PITCH, font_px=30, n_pages=2)   # page 0 shown 0.1 s
    for k in range(60):
        s.write("gaze", t=1.1 + k / 30, x=500.0, y=133.0, valid=True)
    s.write("reading_end", t=3.2)
    s.close()
    _, rows = session_tables(read_events(tmp_path / "k" / "events.jsonl"), CFG, "k")
    by = {r["para_id"]: r for r in rows}
    assert by[0]["page_not_shown"] == 1 and by[1]["page_not_shown"] == 0
