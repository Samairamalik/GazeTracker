"""Live app: webcam -> features -> calibration map -> filter -> GazePoints over WebSocket.

Run with `make app` (or `python -m gazeread.app.server --open`). Python owns the clock:
it tells the page where to draw each calibration dot and collects features on
time.monotonic(), so no browser/Python clock sync is needed.

`--source synthetic` replaces the webcam with fake features whose gaze follows the dot
plus Gaussian noise. It exists to test the app end-to-end without a camera; synthetic
sessions are marked in the log and are never accuracy results.

Backends (spec 3.2): "landmark" (Method A) and "l2cs" (Method B, L2CS-Net on the MediaPipe face
crop). Calibration variants: A (landmark features), B (L2CS gaze yaw/pitch) and B+pose (plus
MediaPipe head yaw/pitch). With gaze.compare, every frame runs both backends and one calibration
fits and validates all three variants on the same dots (a paired comparison); the cursor follows
the primary variant.
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
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import uvicorn
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from gazeread.analysis.heatmap import analyze_session, export
from gazeread.app.session import SessionLog, read_events
from gazeread.gaze.backend_landmark import FEATURE_NAMES, FEATURE_SETS, LandmarkBackend
from gazeread.gaze.backend_l2cs import L2CS_NAMES, L2CS_SETS
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


@dataclass
class Variant:
    name: str            # "A", "B" or "B+pose"
    backend: str         # "landmark" or "l2cs"
    features: str        # feature-set name
    sel: list[int]       # columns of that backend's FrameFeatures.vec


def make_variants(cfg: dict) -> tuple[list[Variant], str]:
    """Active calibration variants and the primary one's name (it drives the cursor)."""
    fa, fb = cfg["calib"]["features"], cfg["l2cs"]["features"]
    a = Variant("A", "landmark", fa, FEATURE_SETS[fa])
    b = Variant("B", "l2cs", "gaze", L2CS_SETS["gaze"])
    bp = Variant("B+pose", "l2cs", "gaze_pose", L2CS_SETS["gaze_pose"])
    primary = a if cfg["gaze"]["backend"] == "landmark" else (bp if fb == "gaze_pose" else b)
    return ([a, b, bp] if cfg["gaze"]["compare"] else [primary]), primary.name


def feature_names(v: Variant) -> list[str]:
    names = FEATURE_NAMES if v.backend == "landmark" else L2CS_NAMES
    return [names[k] for k in v.sel]


class SyntheticSource:
    """TEST ONLY. Gaze jumps to the current target after 200 ms, else drifts on a slow Lissajous;
    `noise_px` Gaussian noise per axis; ~2% of frames invalid (blinks). Features are a fixed
    linear function of the noisy gaze, so a perfect affine calibration gives sigma ~ noise*sqrt(2)."""

    def __init__(self, noise_px: float, seed: int = 0):
        self.noise_px = noise_px
        self.rng = np.random.default_rng(seed)
        self._target, self._t_change, self._g = None, 0.0, None

    def read(self, t: float, target, w: float, h: float) -> dict[str, FrameFeatures]:
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
        valid = bool(self.rng.random() > 0.02)
        bx, by = gx + self.rng.normal(0, self.noise_px / 2), gy + self.rng.normal(0, self.noise_px / 2)
        vb = np.array([(bx / w - 0.5) * 40, -(by / h - 0.5) * 25, *self.rng.normal(0, 0.5, 2)])
        return {"landmark": FrameFeatures(t=t, valid=valid, vec=vec),
                "l2cs": FrameFeatures(t=t, valid=valid, vec=vb)}


class Pipeline:
    """Capture/feature thread plus the state shared with the WebSocket handler."""

    def __init__(self, cfg: dict, source: str):
        self.cfg, self.source = cfg, source
        self.variants, self.primary = make_variants(cfg)
        self.backends = sorted({v.backend for v in self.variants})
        self.buffer: deque[tuple[float, dict[str, FrameFeatures]]] = deque(maxlen=600)   # ~20 s at 30 fps
        self.lock = threading.Lock()
        self.map = None                        # primary variant's calibration map
        self.filter = make_filter(cfg["filter"])
        self.screen = (1280.0, 800.0)
        self.target = None                     # current dot (synthetic source follows it)
        self.session: SessionLog | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.queue: asyncio.Queue | None = None
        self.fps, self.latency_ms, self.dropped = 0.0, 0.0, 0
        self._stop = threading.Event()
        self._cap = self._backend = self._l2cs = self._synth = None

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
        if "l2cs" in self.backends:
            from gazeread.gaze.backend_l2cs import L2CSBackend
            c = self.cfg["l2cs"]
            self._l2cs = L2CSBackend(str(ROOT / c["weights"]), c["crop_scale"], c["device"])
            log.info("L2CS-Net loaded on %s", self._l2cs.device)

    def _next(self) -> dict[str, FrameFeatures] | None:
        if self._synth is not None:
            time.sleep(1 / 30)
            out = self._synth.read(time.monotonic(), self.target, *self.screen)
            return {b: out[b] for b in self.backends}
        ok, frame = self._cap.read()
        t = time.monotonic()
        if not ok:
            self.dropped += 1
            if self.dropped % 30 == 1:
                log.warning("dropped camera frames: %d", self.dropped)
            return None
        det = self._backend.detect(frame, t)
        out = {"landmark": self._backend.from_detection(det, t)}
        if self._l2cs is not None:
            out["l2cs"] = self._l2cs.from_detection(frame, det, out["landmark"].valid, t)
        return {b: out[b] for b in self.backends}

    def run(self) -> None:
        n, t_stats = 0, time.monotonic()
        while not self._stop.is_set():
            frames = self._next()
            if frames is None:
                continue
            pv = next(v for v in self.variants if v.name == self.primary)
            ff = frames[pv.backend]
            with self.lock:
                self.buffer.append((ff.t, frames))
                m = self.map
            if self.session:
                for b, f in frames.items():
                    self.session.write("features", t=f.t, valid=f.valid, vec=f.vec, backend=b)
            if m is not None:
                x, y = m.predict(ff.vec[pv.sel]) if ff.valid else (math.nan, math.nan)
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

    def window(self, t_on: float, t_off: float) -> list[dict[str, FrameFeatures]]:
        with self.lock:
            return [fr for t, fr in self.buffer if t_on <= t <= t_off]

    def stop(self) -> None:
        self._stop.set()
        if self._cap is not None:
            self._cap.release()


async def run_calibration(p: Pipeline, q: asyncio.Queue) -> None:
    """Show fit dots, fit a map per variant, show validation dots, measure sigma_px per variant
    on the same frames, then start the cursor with the primary variant's map."""
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
    X = {v.name: [] for v in p.variants}
    Y = {v.name: [] for v in p.variants}
    for i, (x, y) in enumerate(fit_targets):
        t_on, t_off, fr = await show(x, y, "fit", i, len(fit_targets))
        for v in p.variants:
            ffs = [f[v.backend] for f in fr]
            if not ffs:
                continue
            agg = aggregate_dot(np.array([f.t for f in ffs]), np.array([f.vec for f in ffs]),
                                np.array([f.valid for f in ffs]), t_on, t_off, *skip, c["min_samples"])
            if agg is not None:
                X[v.name].append(agg[v.sel])
                Y[v.name].append((x, y))

    alphas = np.logspace(*c["alphas_log10"][:2], int(c["alphas_log10"][2]))
    models, infos, failed = {}, {}, {}
    for v in p.variants:
        if len(X[v.name]) < max(len(v.sel) + 2, 6):
            failed[v.name] = f"only {len(X[v.name])} of {len(fit_targets)} dots had enough valid frames"
            continue
        models[v.name], infos[v.name] = select_model(np.array(X[v.name]), np.array(Y[v.name]),
                                                     tuple(c["degrees"]), alphas, c["poly2_margin"])

    preds = {v: [] for v in models}
    targs = {v: [] for v in models}
    dot_med = {v: [] for v in models}
    vt = shuffled(validation_targets(w, h), seed + 1)
    for i, (x, y) in enumerate(vt):
        t_on, t_off, fr = await show(x, y, "validate", i, len(vt))
        for v in p.variants:
            if v.name not in models:
                continue
            ffs = [f[v.backend] for f in fr]
            ffs = [f for f in ffs if f.valid and window_mask(np.array([f.t]), t_on, t_off, *skip)[0]]
            if ffs:
                P = models[v.name].predict(np.array([f.vec[v.sel] for f in ffs]))
                preds[v.name].append(P)
                targs[v.name].append(np.tile([x, y], (len(P), 1)))
                dot_med[v.name].append(float(np.linalg.norm(np.median(P, axis=0) - [x, y])))
            else:
                dot_med[v.name].append(None)
    p.target = None

    results = {}
    for v in p.variants:
        common = {"t": time.monotonic(), "variant": v.name, "backend": v.backend,
                  "primary": v.name == p.primary, "n_points": c["n_points"], "features": v.features,
                  "feature_names": feature_names(v), "order_seed": seed, "source": p.source}
        if v.name not in failed and not preds[v.name]:
            failed[v.name] = "no valid frames during validation"
        if v.name in failed:
            p.session.write("calib_result", ok=False, reason=failed[v.name], **common)
            continue
        err = validation_error(np.vstack(preds[v.name]), np.vstack(targs[v.name]))
        results[v.name] = {"ok": True, **common, "n_fit_dots": len(X[v.name]), **infos[v.name],
                           **err, "dot_median_err_px": dot_med[v.name]}
        p.session.write("calib_result", **results[v.name])
        log.info("calibration %s (%s): degree=%d alpha=%.3g LOO=%.1f px sigma=%.1f px", v.name,
                 v.features, infos[v.name]["degree"], infos[v.name]["alpha"],
                 infos[v.name]["loo_rms_px"], err["sigma_px"])
    if p.primary not in results:
        await q.put({"type": "calib_failed", "reason": failed[p.primary]})
        return
    with p.lock:
        p.filter.reset()
        p.map = models[p.primary]
    compare = [{"variant": n, **{k: r[k] for k in ("sigma_px", "rms_x", "rms_y", "loo_rms_px")}}
               for n, r in results.items()]
    done = {k: val for k, val in results[p.primary].items() if k != "candidates"}
    await q.put({"type": "calib_done", **done, "compare": compare if len(p.variants) > 1 else []})


def analyze_and_export(p: Pipeline) -> dict:
    """Analyse the latest reading from the session log itself (same path as offline replay)."""
    p.session.flush()
    result = analyze_session(read_events(p.session.path), p.cfg["analysis"])
    export(result, p.session.dir)
    log.info("analysis: %d pages, granularity %s, written to %s",
             len(result["pages"]), result["granularity"], p.session.dir)
    return result


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
        await q.put({"type": "config", "reading": p.cfg["reading"], "session": p.session.id})
        try:
            while True:
                msg = await ws.receive_json()
                if msg.get("type") == "page":          # page shown now; Python stamps the time
                    p.session.write("page", page=msg["page"], t_on=time.monotonic(),
                                    lines=msg["lines"], line_pitch=msg["line_pitch"],
                                    font_px=msg["font_px"], n_pages=msg["n_pages"])
                elif msg.get("type") == "reading_end":
                    p.session.write("reading_end", t=time.monotonic())
                    try:
                        result = await asyncio.to_thread(analyze_and_export, p)
                        await q.put({"type": "analysis", "result": result,
                                     "dir": str(p.session.dir.relative_to(ROOT))})
                    except ValueError as e:
                        await q.put({"type": "analysis_failed", "reason": str(e)})
                if msg.get("type") == "hello":       # (re)start calibration for this viewport
                    p.screen = (float(msg["w"]), float(msg["h"]))
                    p.session.write("meta", screen_w=msg["w"], screen_h=msg["h"], dpr=msg["dpr"],
                                    fps=p.fps, source=p.source, config_sha256=cfg_hash,
                                    n_points=p.cfg["calib"]["n_points"],
                                    features=p.cfg["calib"]["features"], backend=p.primary,
                                    compare=p.cfg["gaze"]["compare"],
                                    variants=[v.name for v in p.variants])
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
    ap.add_argument("--backend", choices=["landmark", "l2cs"])
    ap.add_argument("--compare", action="store_true", help="run both backends; calibrate A, B, B+pose")
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
    if args.backend:
        cfg["gaze"]["backend"] = args.backend
    if args.compare:
        cfg["gaze"]["compare"] = True
    cfg_hash = hashlib.sha256(yaml.safe_dump(cfg, sort_keys=True).encode()).hexdigest()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print(CONSENT)
    p = Pipeline(cfg, cfg["app"]["source"])
    try:
        p.open()
    except RuntimeError as e:
        sys.exit(str(e))
    p.session = SessionLog(ROOT / "sessions")
    log.info("source=%s variants=%s primary=%s n_points=%d session=%s", p.source,
             [v.name for v in p.variants], p.primary, cfg["calib"]["n_points"], p.session.path)
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
