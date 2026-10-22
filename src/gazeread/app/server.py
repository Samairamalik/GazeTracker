"""Live app: webcam -> features -> calibration map -> filter -> GazePoints over WebSocket.

Run with `make app` (or `python -m gazeread.app.server --open`). Python owns the clock:
it tells the page where to draw each calibration dot and collects features on
time.monotonic(), so no browser/Python clock sync is needed.

`--source synthetic` replaces the webcam with fake features whose gaze follows the dot
plus Gaussian noise. It exists to test the app end-to-end without a camera; synthetic
sessions are marked in the log and are never accuracy results.
"""
import argparse
import asyncio
import copy
import hashlib
import logging
import math
import sys
import threading
import time
import webbrowser
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import uvicorn
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from gazeread.app.session import SessionLog
from gazeread.gaze.backend_landmark import FEATURE_NAMES, FEATURE_SETS, LandmarkBackend
from gazeread.gaze.calibration import (
    aggregate_dot, grid_targets, select_model, shuffled, validation_error, validation_targets,
    window_mask,
)
from gazeread.gaze.filters import make_filter
from gazeread.types import FrameFeatures

ROOT = Path(__file__).resolve().parents[3]
STATIC = Path(__file__).parent / "static"
CONSENT = "Webcam is used live for gaze estimation only; no video or images are saved."
CAMERA_HELP = ("Camera failed to open. On macOS: System Settings > Privacy & Security > Camera, "
               "enable the app running this command (e.g. Terminal), then restart that app.")
log = logging.getLogger("gazeread")


class SyntheticSource:
    """TEST ONLY. Gaze jumps to the current target after 200 ms, else drifts on a slow Lissajous;
    `noise_px` Gaussian noise per axis; ~2% of frames invalid (blinks). Features are a fixed
    linear function of the noisy gaze, so a perfect affine calibration gives sigma ~ noise*sqrt(2)."""

    def __init__(self, noise_px: float, seed: int = 0):
        self.noise_px = noise_px
        self.rng = np.random.default_rng(seed)
        self._target, self._t_change, self._g = None, 0.0, None

    def read(self, t: float, target, w: float, h: float) -> FrameFeatures:
        if target != self._target:
            self._target, self._t_change = target, t
        if target is not None and t - self._t_change >= 0.2:
            self._g = np.array(target, float)
        elif target is None:
            self._g = np.array([w * (0.5 + 0.35 * math.sin(0.4 * t)), h * (0.5 + 0.35 * math.sin(0.27 * t))])
        elif self._g is None:
            self._g = np.array([w / 2, h / 2])
        gx, gy = self._g + self.rng.normal(0, self.noise_px, 2)
        u, v = (gx / w - 0.5) * 0.4, (gy / h - 0.5) * 0.3
        vec = np.array([u, v, -u, v, 0.3, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 60.0])
        return FrameFeatures(t=t, valid=bool(self.rng.random() > 0.02), vec=vec)


class Pipeline:
    """Capture/feature thread plus the state shared with the WebSocket handler."""

    def __init__(self, cfg: dict, source: str):
        self.cfg, self.source = cfg, source
        self.sel = FEATURE_SETS[cfg["calib"]["features"]]
        self.buffer: deque[FrameFeatures] = deque(maxlen=600)    # ~20 s at 30 fps
        self.lock = threading.Lock()
        self.map = None
        self.filter = make_filter(cfg["filter"])
        self.screen = (1280.0, 800.0)
        self.target = None                     # current dot (synthetic source follows it)
        self.session: SessionLog | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.queue: asyncio.Queue | None = None
        self.fps, self.latency_ms, self.dropped = 0.0, 0.0, 0
        self._stop = threading.Event()
        self._cap = self._backend = self._synth = None

    def open(self) -> None:
        if self.source == "synthetic":
            self._synth = SyntheticSource(self.cfg["app"]["synthetic_noise_px"])
            return
        self._cap = cv2.VideoCapture(self.cfg["camera"]["index"], cv2.CAP_AVFOUNDATION)
        if not (self._cap.isOpened() and self._cap.read()[0]):
            raise RuntimeError(CAMERA_HELP)
        lm, pose = self.cfg["landmarks"], self.cfg["pose"]
        self._backend = LandmarkBackend(str(ROOT / lm["model_path"]), lm["eye_open_min"],
                                        pose["max_yaw_deg"], pose["max_pitch_deg"], lm["num_faces"])

    def _next(self) -> FrameFeatures | None:
        if self._synth is not None:
            time.sleep(1 / 30)
            return self._synth.read(time.monotonic(), self.target, *self.screen)
        ok, frame = self._cap.read()
        t = time.monotonic()
        if not ok:
            self.dropped += 1
            if self.dropped % 30 == 1:
                log.warning("dropped camera frames: %d", self.dropped)
            return None
        return self._backend.process(frame, t)

    def run(self) -> None:
        n, t_stats = 0, time.monotonic()
        while not self._stop.is_set():
            ff = self._next()
            if ff is None:
                continue
            with self.lock:
                self.buffer.append(ff)
                m = self.map
            if self.session:
                self.session.write("features", t=ff.t, valid=ff.valid, vec=ff.vec)
            if m is not None:
                x, y = m.predict(ff.vec[self.sel]) if ff.valid else (math.nan, math.nan)
                gp = self.filter(ff.t, float(x), float(y), ff.valid)
                if self.session:
                    self.session.write("gaze", t=gp.t, x=gp.x, y=gp.y, valid=gp.valid)
                self.emit({"type": "gaze", "t": gp.t, "x": gp.x if gp.valid else None,
                           "y": gp.y if gp.valid else None, "valid": gp.valid})
            n += 1
            now = time.monotonic()
            if now - t_stats >= 5.0:
                self.fps, n, t_stats = n / (now - t_stats), 0, now
                log.info("fps %.1f  pipeline latency %.1f ms", self.fps, self.latency_ms)
                self.emit({"type": "stats", "fps": self.fps, "latency_ms": self.latency_ms})

    def emit(self, msg: dict) -> None:
        """Thread-safe enqueue for the WebSocket sender; drops the oldest message if the page lags."""
        loop, q = self.loop, self.queue
        if loop is None or q is None:
            return

        def put():
            if q.full():
                q.get_nowait()
            q.put_nowait(msg)
        loop.call_soon_threadsafe(put)

    def window(self, t_on: float, t_off: float) -> list[FrameFeatures]:
        with self.lock:
            return [f for f in self.buffer if t_on <= f.t <= t_off]

    def stop(self) -> None:
        self._stop.set()
        if self._cap is not None:
            self._cap.release()


async def run_calibration(p: Pipeline, q: asyncio.Queue) -> None:
    """Show fit dots, fit the map, show validation dots, measure sigma_px, then start the cursor."""
    c = p.cfg["calib"]
    w, h = p.screen
    skip = (c["skip_start_s"], c["skip_end_s"])
    with p.lock:
        p.map = None

    async def show(x, y, phase, i, n):
        p.target = (float(x), float(y))
        await q.put({"type": "dot", "x": float(x), "y": float(y), "phase": phase, "i": i, "n": n,
                     "dur": c["dot_s"]})
        t_on = time.monotonic()
        await asyncio.sleep(c["dot_s"])
        t_off = time.monotonic()
        p.session.write("calib_target", x=x, y=y, t_on=t_on, t_off=t_off, phase=phase)
        return t_on, t_off, p.window(t_on, t_off)

    seed = int(time.time_ns() % 2**31) if c["order_seed"] is None else int(c["order_seed"])
    fit_targets = shuffled(grid_targets(c["n_points"], w, h, c["inset"]), seed)
    X, Y = [], []
    for i, (x, y) in enumerate(fit_targets):
        t_on, t_off, fr = await show(x, y, "fit", i, len(fit_targets))
        if not fr:
            continue
        agg = aggregate_dot(np.array([f.t for f in fr]), np.array([f.vec for f in fr]),
                            np.array([f.valid for f in fr]), t_on, t_off, *skip, c["min_samples"])
        if agg is not None:
            X.append(agg[p.sel])
            Y.append((x, y))
    if len(X) < max(len(p.sel) + 2, 6):
        p.target = None
        reason = f"only {len(X)} of {len(fit_targets)} dots had enough valid frames"
        p.session.write("calib_result", ok=False, reason=reason)
        await q.put({"type": "calib_failed", "reason": reason})
        return
    alphas = np.logspace(*c["alphas_log10"][:2], int(c["alphas_log10"][2]))
    model, info = select_model(np.array(X), np.array(Y), tuple(c["degrees"]), alphas, c["poly2_margin"])

    preds, targs, dot_med = [], [], []
    vt = shuffled(validation_targets(w, h), seed + 1)
    for i, (x, y) in enumerate(vt):
        t_on, t_off, fr = await show(x, y, "validate", i, len(vt))
        fr = [f for f in fr if f.valid and window_mask(np.array([f.t]), t_on, t_off, *skip)[0]]
        if fr:
            P = model.predict(np.array([f.vec[p.sel] for f in fr]))
            preds.append(P)
            targs.append(np.tile([x, y], (len(P), 1)))
            dot_med.append(float(np.linalg.norm(np.median(P, axis=0) - [x, y])))
        else:
            dot_med.append(None)
    p.target = None
    if not preds:
        reason = "no valid frames during validation"
        p.session.write("calib_result", ok=False, reason=reason)
        await q.put({"type": "calib_failed", "reason": reason})
        return
    err = validation_error(np.vstack(preds), np.vstack(targs))
    result = {"ok": True, "n_points": c["n_points"], "features": c["features"],
              "feature_names": [FEATURE_NAMES[k] for k in p.sel], "n_fit_dots": len(X),
              "order_seed": seed,
              **info, **err, "dot_median_err_px": dot_med, "source": p.source}
    p.session.write("calib_result", **result)
    log.info("calibration: %s degree=%d alpha=%.3g LOO=%.1f px sigma=%.1f px",
             c["features"], info["degree"], info["alpha"], info["loo_rms_px"], err["sigma_px"])
    with p.lock:
        p.filter.reset()
        p.map = model
    await q.put({"type": "calib_done", **{k: v for k, v in result.items() if k != "candidates"}})


def create_app(p: Pipeline, cfg_hash: str) -> FastAPI:
    app = FastAPI()

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=16)
        p.loop, p.queue = asyncio.get_running_loop(), q

        async def sender():
            while True:
                msg = await q.get()
                if msg.get("type") == "gaze":
                    p.latency_ms = 0.9 * p.latency_ms + 0.1 * (time.monotonic() - msg["t"]) * 1000
                await ws.send_json(msg)

        send_task, calib_task = asyncio.create_task(sender()), None
        try:
            while True:
                msg = await ws.receive_json()
                if msg.get("type") == "hello":       # (re)start calibration for this viewport
                    p.screen = (float(msg["w"]), float(msg["h"]))
                    p.session.write("meta", screen_w=msg["w"], screen_h=msg["h"], dpr=msg["dpr"],
                                    fps=p.fps, source=p.source, config_sha256=cfg_hash,
                                    n_points=p.cfg["calib"]["n_points"],
                                    features=p.cfg["calib"]["features"])
                    if calib_task:
                        calib_task.cancel()
                    calib_task = asyncio.create_task(run_calibration(p, q))
        except WebSocketDisconnect:
            pass
        finally:
            for task in (calib_task, send_task):
                if task:
                    task.cancel()
            p.queue = None
            with p.lock:
                p.map = None

    app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--source", choices=["camera", "synthetic"])
    ap.add_argument("--n-points", type=int, choices=[9, 16])
    ap.add_argument("--features", choices=sorted(FEATURE_SETS))
    ap.add_argument("--port", type=int)
    ap.add_argument("--open", action="store_true", help="open the page in the default browser")
    args = ap.parse_args()

    cfg = copy.deepcopy(yaml.safe_load(Path(args.config).read_text()))
    for key, val in (("source", args.source), ("port", args.port)):
        if val is not None:
            cfg["app"][key] = val
    if args.n_points:
        cfg["calib"]["n_points"] = args.n_points
    if args.features:
        cfg["calib"]["features"] = args.features
    cfg_hash = hashlib.sha256(yaml.safe_dump(cfg, sort_keys=True).encode()).hexdigest()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(CONSENT)
    p = Pipeline(cfg, cfg["app"]["source"])
    try:
        p.open()
    except RuntimeError as e:
        sys.exit(str(e))
    p.session = SessionLog(ROOT / "sessions")
    log.info("source=%s n_points=%d features=%s session=%s", p.source, cfg["calib"]["n_points"],
             cfg["calib"]["features"], p.session.path)
    threading.Thread(target=p.run, daemon=True).start()
    url = f"http://{cfg['app']['host']}:{cfg['app']['port']}"
    if args.open:
        threading.Timer(1.0, webbrowser.open, [url]).start()
    log.info("open %s", url)
    try:
        uvicorn.run(create_app(p, cfg_hash), host=cfg["app"]["host"], port=cfg["app"]["port"],
                    log_level="warning")
    finally:
        p.stop()
        p.session.close()


if __name__ == "__main__":
    main()
