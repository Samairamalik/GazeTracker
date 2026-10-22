"""Session log: JSONL, one event per line, in sessions/<id>/events.jsonl (spec 4.3).

Holds landmark-derived features, gaze coordinates and events only, never images.
Thread-safe; buffered with a flush at most every `flush_s` seconds.
"""
import json
import math
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np


def _clean(v):
    """Make a value JSON-safe: numpy -> python, NaN/inf -> None."""
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_clean(x) for x in (v.tolist() if isinstance(v, np.ndarray) else v)]
    if isinstance(v, (np.floating, float)):
        v = float(v)
        return v if math.isfinite(v) else None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


class SessionLog:
    def __init__(self, root: str | Path, session_id: str | None = None, flush_s: float = 1.0):
        self.id = session_id or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = Path(root) / self.id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "events.jsonl"
        self._f = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self._flush_s, self._last_flush = flush_s, time.monotonic()

    def write(self, type_: str, **fields) -> None:
        line = json.dumps({"type": type_, **_clean(fields)}, allow_nan=False)
        with self._lock:
            self._f.write(line + "\n")
            now = time.monotonic()
            if now - self._last_flush >= self._flush_s:
                self._f.flush()
                self._last_flush = now

    def flush(self) -> None:
        with self._lock:
            self._f.flush()
            self._last_flush = time.monotonic()

    def close(self) -> None:
        with self._lock:
            self._f.close()


def read_events(path: str | Path) -> list[dict]:
    """All events of a session log, in order."""
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
