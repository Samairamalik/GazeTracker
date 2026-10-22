import numpy as np

from gazeread.app.session import SessionLog, read_events


def test_jsonl_roundtrip(tmp_path):
    s = SessionLog(tmp_path, session_id="t1")
    s.write("meta", screen_w=1440, screen_h=900, dpr=2.0, source="synthetic")
    s.write("features", t=1.5, valid=np.bool_(True), vec=np.array([0.1, np.nan, 3.0]))
    s.write("gaze", t=1.5, x=np.float32(10.5), y=float("inf"), valid=False)
    s.close()
    ev = read_events(tmp_path / "t1" / "events.jsonl")
    assert [e["type"] for e in ev] == ["meta", "features", "gaze"]
    assert ev[0]["source"] == "synthetic"
    assert ev[1]["vec"] == [0.1, None, 3.0] and ev[1]["valid"] is True
    assert ev[2]["x"] == 10.5 and ev[2]["y"] is None
