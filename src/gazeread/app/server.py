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
import base64
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
from gazeread.app.quiz import load_questions, public_questions, score
from gazeread.app.session import SessionLog, read_events
from gazeread.features.landmarks import extract_hud_mesh, synthetic_hud_mesh
from gazeread.gaze.backend_landmark import FEATURE_NAMES, FEATURE_SETS, LandmarkBackend
from gazeread.gaze.backend_l2cs import L2CS_NAMES, L2CS_SETS
from gazeread.gaze.calibration import (
    aggregate_dot, grid_targets, loo_residuals, select_model, shuffled,
    validation_error, validation_targets, window_mask,
)
from gazeread.gaze.filters import make_filter
from gazeread.types import FrameFeatures

ROOT = Path(__file__).resolve().parents[3]
STATIC = Path(__file__).parent / "static"
CONSENT = "Webcam is used live for gaze estimation only; no video or images are saved."
CAMERA_HELP = (
    "Camera failed to open.\n"
    "- Windows: Settings > Privacy & security > Camera (allow desktop apps), or close other apps using the webcam.\n"
    "- macOS: System Settings > Privacy & Security > Camera, enable your terminal app, then restart it.\n"
    "- Linux: check /dev/video0 permissions (video group).\n"
    "- No webcam? Run with `--source synthetic` to use the interactive synthetic biometric simulator."
)
log = logging.getLogger("gazeread")


def open_camera(index: int) -> cv2.VideoCapture:
    """Cross-platform webcam opener: tries platform-native backends and configures 640x480 @ 30fps MJPG."""
    if sys.platform == "darwin":
        backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    elif sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif sys.platform.startswith("linux"):
        backends = [getattr(cv2, "CAP_V4L2", cv2.CAP_ANY), cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]

    for api in backends:
        try:
            cap = cv2.VideoCapture(index, api)
            if cap is not None and cap.isOpened():
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ok, _ = cap.read()
                if ok:
                    log.info("Opened webcam index=%d with backend=%s", index, api)
                    return cap
                cap.release()
        except Exception:
            pass
    raise RuntimeError(CAMERA_HELP)


def list_passages(texts_dir: Path) -> list[dict]:
    """Metadata for all available reading passages in `static/texts`."""
    out = []
    if not texts_dir.exists():
        return out
    for p in sorted(texts_dir.glob("*.txt")):
        name = p.stem
        raw = p.read_text(encoding="utf-8").strip()
        paras = [ blk.strip() for blk in raw.split("\n\n") if blk.strip() ]
        words = sum(len(blk.split()) for blk in paras)
        qs = load_questions(texts_dir, name)
        q_count = len(qs) if qs else (4 if name == "alice_excerpts" else 0)
        title = name.replace("_", " ").title()
        out.append({
            "id": name,
            "title": title,
            "para_count": len(paras),
            "word_count": words,
            "quiz_count": q_count,
        })
    return out


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

    def read(self, t: float, target, w: float, h: float) -> tuple[dict[str, FrameFeatures], dict]:
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
        valid = bool(self.rng.random() > 0.02)
        open_r = (0.31 + 0.02 * math.sin(1.7 * t)) if valid else 0.04
        open_l = (0.31 + 0.02 * math.cos(1.5 * t)) if valid else 0.04
        yaw = 2.8 * math.sin(0.35 * t)
        pitch = 2.1 * math.cos(0.28 * t)
        roll = 1.1 * math.sin(0.22 * t)
        tx = 0.8 * math.sin(0.25 * t)
        ty = 0.6 * math.cos(0.31 * t)
        tz = 58.5 + 1.8 * math.sin(0.18 * t)
        vec = np.array([u, v, -u, v, open_r, open_l, yaw, pitch, roll, tx, ty, tz])
        bx, by = gx + self.rng.normal(0, self.noise_px / 2), gy + self.rng.normal(0, self.noise_px / 2)
        vb = np.array([(bx / w - 0.5) * 40, -(by / h - 0.5) * 25, *self.rng.normal(0, 0.5, 2)])
        mesh = synthetic_hud_mesh(float(u), float(v), float(-u), float(v), float(open_r), float(open_l), yaw, pitch, roll)
        return (
            {"landmark": FrameFeatures(t=t, valid=valid, vec=vec),
             "l2cs": FrameFeatures(t=t, valid=valid, vec=vb)},
            mesh,
        )


class Pipeline:
    """Capture/feature thread plus the state shared with the WebSocket handler."""

    def __init__(self, cfg: dict, source: str):
        self.cfg, self.source = cfg, source
        self.variants, self.primary = make_variants(cfg)
        self.backends = sorted({v.backend for v in self.variants})
        self.buffer: deque[tuple[float, dict[str, FrameFeatures]]] = deque(maxlen=900)   # ~30 s at 30 fps
        self.lock = threading.Lock()
        self.map = None                        # primary variant's calibration map
        self.filter = make_filter(cfg["filter"])
        self.screen = (1280.0, 800.0)
        self.target = None                     # current dot (synthetic source follows it)
        self.session: SessionLog | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.queue: asyncio.Queue | None = None
        self.fps, self.latency_ms, self.dropped = 0.0, 0.0, 0
        self.logged_t = -math.inf              # capture time of the last frame fully written to the log
        self.drift_offset = np.zeros(2, float) # micro-drift correction offset (dx, dy) in px
        self.drift_lock_enabled = True
        self._last_mesh = None
        self._last_preview = None
        self._stop = threading.Event()
        self._cap = self._backend = self._l2cs = self._synth = None

    def open(self) -> None:
        if self.source == "synthetic":
            self._synth = SyntheticSource(self.cfg["app"]["synthetic_noise_px"])
            return
        self._cap = open_camera(self.cfg["camera"]["index"])
        lm, pose = self.cfg["landmarks"], self.cfg["pose"]
        self._backend = LandmarkBackend(str(ROOT / lm["model_path"]), lm["eye_open_min"],
                                        pose["max_yaw_deg"], pose["max_pitch_deg"], lm["num_faces"])
        if "l2cs" in self.backends:
            from gazeread.gaze.backend_l2cs import L2CSBackend
            c = self.cfg["l2cs"]
            self._l2cs = L2CSBackend(str(ROOT / c["weights"]), c["crop_scale"], c["device"])
            log.info("L2CS-Net loaded on %s", self._l2cs.device)

    def rebuild_variants(self) -> None:
        with self.lock:
            self.variants, self.primary = make_variants(self.cfg)
            self.backends = sorted({v.backend for v in self.variants})

    def _next(self) -> dict[str, FrameFeatures] | None:
        if self._synth is not None:
            time.sleep(1 / 30)
            out, mesh = self._synth.read(time.monotonic(), self.target, *self.screen)
            self._last_mesh = mesh
            self._last_preview = None
            return {b: out[b] for b in self.backends}
        ok, frame = self._cap.read()
        t = time.monotonic()
        if not ok:
            self.dropped += 1
            if self.dropped % 30 == 1:
                log.warning("dropped camera frames: %d", self.dropped)
            return None
        det = self._backend.detect(frame, t)
        self._last_mesh = extract_hud_mesh(det)
        if self.map is None and self.target is None:
            small = cv2.resize(cv2.flip(frame, 1), (320, 220))
            ok_jpg, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 55])
            self._last_preview = base64.b64encode(buf).decode("ascii") if ok_jpg else None
        else:
            self._last_preview = None
        out = {"landmark": self._backend.from_detection(det, t)}
        if self._l2cs is not None:
            out["l2cs"] = self._l2cs.from_detection(frame, det, out["landmark"].valid, t)
        return {b: out[b] for b in self.backends}

    def run(self) -> None:
        n, t_stats = 0, time.monotonic()
        t_fps_win = deque(maxlen=35)
        while not self._stop.is_set():
            frames = self._next()
            if frames is None:
                continue
            pv = next(v for v in self.variants if v.name == self.primary)
            ff = frames[pv.backend]
            t_fps_win.append(ff.t)
            if len(t_fps_win) >= 4:
                dt_win = t_fps_win[-1] - t_fps_win[0]
                if dt_win > 1e-3:
                    self.fps = (len(t_fps_win) - 1) / dt_win
            with self.lock:
                self.buffer.append((ff.t, frames))
                m = self.map
                drift = self.drift_offset.copy() if self.drift_lock_enabled else np.zeros(2)
            if self.session:
                for b, f in frames.items():
                    self.session.write("features", t=f.t, valid=f.valid, vec=f.vec, backend=b)
            if m is not None:
                if ff.valid:
                    pred = m.predict(ff.vec[pv.sel])
                    x, y = float(pred[0] + drift[0]), float(pred[1] + drift[1])
                else:
                    x, y = math.nan, math.nan
                gp = self.filter(ff.t, float(x), float(y), ff.valid)
                if self.session:
                    self.session.write("gaze", t=gp.t, x=gp.x, y=gp.y, valid=gp.valid)
                self.emit({"type": "gaze", "t": gp.t, "x": gp.x if gp.valid else None,
                           "y": gp.y if gp.valid else None, "valid": gp.valid})

            # Stream live Biometric HUD telemetry at ~15 Hz
            if n % 2 == 0 and "landmark" in frames:
                lf = frames["landmark"]
                v = lf.vec
                fin = lambda val: round(float(val), 4) if math.isfinite(float(val)) else None
                self.emit({
                    "type": "telemetry",
                    "t": round(lf.t, 3),
                    "valid": bool(lf.valid),
                    "source": self.source,
                    "fps": round(self.fps, 1),
                    "latency_ms": round(self.latency_ms, 1),
                    "uR": fin(v[0]), "vR": fin(v[1]),
                    "uL": fin(v[2]), "vL": fin(v[3]),
                    "openR": fin(v[4]), "openL": fin(v[5]),
                    "yaw": fin(v[6]), "pitch": fin(v[7]), "roll": fin(v[8]),
                    "tx": fin(v[9]), "ty": fin(v[10]), "tz": fin(v[11]),
                    "mesh": self._last_mesh,
                    "preview": self._last_preview,
                })

            self.logged_t = ff.t
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
        p.drift_offset = np.zeros(2, float)

    async def show(x, y, phase, i, n):
        p.target = (float(x), float(y))
        dur = float(c["dot_s"])
        await q.put({"type": "dot", "x": float(x), "y": float(y), "phase": phase, "i": i, "n": n,
                     "dur": dur})
        t_on = time.monotonic()
        # Stream live blink-aware lock progress at ~15 Hz; extend up to +0.9s if blinks paused collection
        pv = next(v for v in p.variants if v.name == p.primary)
        target_valid = max(int(c["min_samples"]) + 4, int((dur - skip[0] - skip[1]) * 24))
        elapsed = 0.0
        extra_budget = 0.9
        step_s = 0.065
        while elapsed < dur:
            await asyncio.sleep(step_s)
            now = time.monotonic()
            fr_so_far = p.window(t_on, now)
            ffs = [f[pv.backend] for f in fr_so_far]
            latest_valid = bool(ffs[-1].valid) if ffs else True
            n_valid = sum(1 for f in ffs if f.valid and f.t >= t_on + skip[0])
            if not latest_valid and extra_budget > 0 and elapsed >= skip[0]:
                # Pause timer advance during blink so user doesn't record bad samples
                extra_budget -= step_s
            else:
                elapsed += step_s
            prog = min(1.0, max(0.0, elapsed / dur))
            await q.put({
                "type": "dot_progress",
                "phase": phase,
                "i": i,
                "n": n,
                "progress": round(prog, 3),
                "locked": latest_valid,
                "valid_samples": n_valid,
                "target_samples": target_valid,
            })
        t_off = time.monotonic()
        p.session.write("calib_target", x=x, y=y, t_on=t_on, t_off=t_off, phase=phase)
        return t_on, t_off, p.window(t_on, t_off)

    seed = int(time.time_ns() % 2**31) if c["order_seed"] is None else int(c["order_seed"])
    fit_targets = shuffled(grid_targets(c["n_points"], w, h, c["inset"]), seed)
    X = {v.name: [] for v in p.variants}
    Y = {v.name: [] for v in p.variants}
    retry_queue = []

    for i, (x, y) in enumerate(fit_targets):
        t_on, t_off, fr = await show(x, y, "fit", i, len(fit_targets))
        pv_ok = False
        for v in p.variants:
            ffs = [f[v.backend] for f in fr]
            if not ffs:
                continue
            agg = aggregate_dot(np.array([f.t for f in ffs]), np.array([f.vec for f in ffs]),
                                np.array([f.valid for f in ffs]), t_on, t_off, *skip, c["min_samples"])
            if agg is not None:
                X[v.name].append(agg[v.sel])
                Y[v.name].append((x, y))
                if v.name == p.primary:
                    pv_ok = True
        if not pv_ok:
            retry_queue.append((i, x, y))

    # Automatic single-dot retry if any dot was missed due to blinks/head movement
    for retry_idx, (orig_i, x, y) in enumerate(retry_queue[:2]):
        t_on, t_off, fr = await show(x, y, "retry", retry_idx, len(retry_queue[:2]))
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
    fit_Transfer = {}
    for v in p.variants:
        if len(X[v.name]) < max(len(v.sel) + 2, 6):
            failed[v.name] = f"only {len(X[v.name])} of {len(fit_targets)} dots had enough valid frames"
            continue
        arr_x, arr_y = np.array(X[v.name]), np.array(Y[v.name])
        models[v.name], infos[v.name] = select_model(arr_x, arr_y,
                                                     tuple(c["degrees"]), alphas, c["poly2_margin"])
        loo_res = loo_residuals(arr_x, arr_y, infos[v.name]["degree"], infos[v.name]["alpha"], models[v.name].scaler)
        fit_Transfer[v.name] = [
            {
                "x": round(float(arr_y[k, 0]), 1),
                "y": round(float(arr_y[k, 1]), 1),
                "pred_x": round(float(arr_y[k, 0] - loo_res[k, 0]), 1) if np.isfinite(loo_res[k, 0]) else None,
                "pred_y": round(float(arr_y[k, 1] - loo_res[k, 1]), 1) if np.isfinite(loo_res[k, 1]) else None,
                "err_px": round(float(np.hypot(loo_res[k, 0], loo_res[k, 1])), 1) if np.all(np.isfinite(loo_res[k])) else None,
            }
            for k in range(len(arr_y))
        ]

    preds = {v: [] for v in models}
    targs = {v: [] for v in models}
    dot_med = {v: [] for v in models}
    val_details = {v: [] for v in models}
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
                med = np.median(P, axis=0)
                med_err = float(np.linalg.norm(med - [x, y]))
                dot_med[v.name].append(med_err)
                val_details[v.name].append({
                    "x": round(float(x), 1),
                    "y": round(float(y), 1),
                    "pred_x": round(float(med[0]), 1),
                    "pred_y": round(float(med[1]), 1),
                    "err_px": round(med_err, 1),
                    "samples": [[round(float(pt[0]), 1), round(float(pt[1]), 1)] for pt in P[::max(1, len(P)//12)]],
                })
            else:
                dot_med[v.name].append(None)
                val_details[v.name].append({
                    "x": round(float(x), 1), "y": round(float(y), 1),
                    "pred_x": None, "pred_y": None, "err_px": None, "samples": [],
                })
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
    await q.put({
        "type": "calib_done",
        **done,
        "compare": compare if len(p.variants) > 1 else [],
        "validation_points": val_details.get(p.primary, []),
        "fit_points": fit_Transfer.get(p.primary, []),
        "line_pitch": p.cfg["reading"]["font_px"] * p.cfg["reading"]["line_height"],
    })


def wait_until_logged(p, t: float, timeout_s: float = 1.0) -> bool:
    """Block until the capture thread has logged a frame captured after `t`, so every frame
    captured before `t` is already in the log. False on timeout (e.g. the camera stalled)."""
    deadline = time.monotonic() + timeout_s
    while p.logged_t <= t:
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.005)
    return True


def analyze_and_export(p: Pipeline, t_end: float) -> dict:
    """Analyse the latest reading from the session log itself (same path as offline replay).
    Waits for in-flight frames captured before `t_end` so the replay sees the same events."""
    if not wait_until_logged(p, t_end):
        log.warning("no frame logged after reading_end within 1 s; analysis may miss in-flight samples")
    p.session.flush()
    result = analyze_session(read_events(p.session.path), p.cfg["analysis"])
    export(result, p.session.dir)
    log.info("analysis: %d pages, granularity %s, written to %s",
             len(result["pages"]), result["granularity"], p.session.dir)
    return result


async def apply_micro_drift_lock(p: Pipeline, anchor_x: float, anchor_y: float, dur_s: float = 0.35) -> None:
    """Zero-effort micro-drift lock: during the 350ms page-start anchor on the first word,
    nudge `p.drift_offset` toward `(anchor_x, anchor_y)` if valid samples are near the anchor."""
    if not p.drift_lock_enabled:
        return
    t_on = time.monotonic()
    await asyncio.sleep(dur_s)
    t_off = time.monotonic()
    with p.lock:
        m = p.map
        pv = next(v for v in p.variants if v.name == p.primary)
    if m is None:
        return
    fr = p.window(t_on, t_off)
    ffs = [f[pv.backend] for f in fr if f[pv.backend].valid]
    if len(ffs) < 3:
        return
    preds = m.predict(np.array([f.vec[pv.sel] for f in ffs]))
    med = np.median(preds, axis=0)
    raw_err = np.array([anchor_x, anchor_y], float) - med
    # Only apply micro-drift correction if the user is actually looking near the top-left start of the page
    if np.linalg.norm(raw_err) <= 180.0:
        with p.lock:
            new_drift = 0.72 * p.drift_offset + 0.28 * raw_err
            norm = np.linalg.norm(new_drift)
            if norm > 55.0:
                new_drift = new_drift * (55.0 / norm)
            p.drift_offset = new_drift
        p.emit({
            "type": "drift_locked",
            "dx": round(float(p.drift_offset[0]), 1),
            "dy": round(float(p.drift_offset[1]), 1),
        })


def ensure_questions(texts_dir: Path, name: str, custom_text: str | None = None) -> list[dict]:
    """Load passage questions from JSON, or generate comprehension questions if none exist on disk."""
    if not custom_text:
        qs = load_questions(texts_dir, name)
        if qs:
            return qs
    raw = custom_text or (texts_dir / f"{name}.txt").read_text(encoding="utf-8")
    paras = [blk.replace("\n", " ").strip() for blk in raw.split("\n\n") if blk.strip()]
    out = []
    for idx, p in enumerate(paras[:4]):
        words = [w.strip(".,!?;:\"'()[]") for w in p.split() if len(w.strip(".,!?;:\"'()[]")) >= 5]
        target = words[len(words) // 2] if words else "detail"
        distractors = ["unrelated concept", "opposite outcome", "secondary footnote"]
        opts = [target, *distractors]
        out.append({
            "id": f"custom_q{idx}",
            "para_id": idx,
            "text": f"In Paragraph {idx + 1}, which key term or detail appeared in the text?",
            "options": opts,
            "answer": 0,
        })
    return out


def create_app(p: Pipeline, cfg_hash: str) -> FastAPI:
    app = FastAPI()

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        q: asyncio.Queue = asyncio.Queue(maxsize=32)
        p.loop, p.queue = asyncio.get_running_loop(), q

        async def sender():
            while True:
                msg = await q.get()
                if msg.get("type") == "gaze":
                    p.latency_ms = 0.9 * p.latency_ms + 0.1 * (time.monotonic() - msg["t"]) * 1000
                await ws.send_json(msg)

        send_task, calib_task, drift_task = asyncio.create_task(sender()), None, None
        questions = ensure_questions(STATIC / "texts", p.cfg["reading"]["text"], p.cfg["reading"].get("custom_text"))
        passages = list_passages(STATIC / "texts")
        await q.put({
            "type": "config",
            "reading": p.cfg["reading"],
            "calib": {"n_points": p.cfg["calib"]["n_points"], "features": p.cfg["calib"]["features"]},
            "source": p.source,
            "session": p.session.id,
            "questions": public_questions(questions),
            "passages": passages,
        })
        try:
            while True:
                msg = await ws.receive_json()
                mtype = msg.get("type")
                if mtype == "configure":
                    # Update passage, calibration grid, cursor visibility, or drift lock from Stage 1 HUD
                    if "text" in msg and msg["text"]:
                        p.cfg["reading"]["text"] = str(msg["text"])
                    if "custom_text" in msg and msg["custom_text"]:
                        p.cfg["reading"]["custom_text"] = str(msg["custom_text"])
                    elif "text" in msg:
                        p.cfg["reading"].pop("custom_text", None)
                    questions = ensure_questions(STATIC / "texts", p.cfg["reading"]["text"], p.cfg["reading"].get("custom_text"))
                    if "show_cursor" in msg:
                        p.cfg["reading"]["show_cursor"] = bool(msg["show_cursor"])
                    if "skim" in msg:
                        p.cfg["reading"]["skim"] = bool(msg["skim"])
                    if "n_points" in msg and int(msg["n_points"]) in (9, 16):
                        p.cfg["calib"]["n_points"] = int(msg["n_points"])
                    if "features" in msg and msg["features"] in FEATURE_SETS:
                        p.cfg["calib"]["features"] = str(msg["features"])
                        p.rebuild_variants()
                    if "drift_lock" in msg:
                        p.drift_lock_enabled = bool(msg["drift_lock"])
                    await q.put({
                        "type": "config",
                        "reading": p.cfg["reading"],
                        "calib": {"n_points": p.cfg["calib"]["n_points"], "features": p.cfg["calib"]["features"]},
                        "source": p.source,
                        "session": p.session.id,
                        "questions": public_questions(questions),
                        "passages": passages,
                    })
                elif mtype == "page":          # page shown now; Python stamps the time
                    p.session.write("page", page=msg["page"], t_on=time.monotonic(),
                                    lines=msg["lines"], line_pitch=msg["line_pitch"],
                                    font_px=msg["font_px"], n_pages=msg["n_pages"])
                    if msg.get("lines") and p.drift_lock_enabled:
                        first_ln = msg["lines"][0]
                        ax = float(first_ln["x0"]) + 24.0
                        ay = 0.5 * (float(first_ln["y0"]) + float(first_ln["y1"]))
                        if drift_task:
                            drift_task.cancel()
                        drift_task = asyncio.create_task(apply_micro_drift_lock(p, ax, ay, 0.35))
                elif mtype == "answer":        # scored here; the page has no answer key
                    para_id, correct = score(questions, msg["question_id"], msg["response"])
                    p.session.write("answer", t=time.monotonic(), passage=p.cfg["reading"]["text"],
                                    question_id=msg["question_id"], para_id=para_id,
                                    response=int(msg["response"]), correct=correct)
                elif mtype == "reading_end":
                    t_end = time.monotonic()
                    p.session.write("reading_end", t=t_end)
                    try:
                        result = await asyncio.to_thread(analyze_and_export, p, t_end)
                        await q.put({"type": "analysis", "result": result,
                                     "dir": str(p.session.dir.relative_to(ROOT)),
                                     "session_id": p.session.id})
                    except ValueError as e:
                        await q.put({"type": "analysis_failed", "reason": str(e)})
                if mtype == "hello":           # (re)start calibration for this viewport
                    p.screen = (float(msg["w"]), float(msg["h"]))
                    p.session.write("meta", screen_w=msg["w"], screen_h=msg["h"], dpr=msg["dpr"],
                                    fps=p.fps, source=p.source, config_sha256=cfg_hash,
                                    n_points=p.cfg["calib"]["n_points"],
                                    features=p.cfg["calib"]["features"], backend=p.primary,
                                    compare=p.cfg["gaze"]["compare"],
                                    variants=[v.name for v in p.variants],
                                    text=p.cfg["reading"]["text"], skim=p.cfg["reading"]["skim"])
                    if calib_task:
                        calib_task.cancel()
                    calib_task = asyncio.create_task(run_calibration(p, q))
        except WebSocketDisconnect:
            pass
        finally:
            for task in (calib_task, drift_task, send_task):
                if task:
                    task.cancel()
            p.queue = None
            with p.lock:
                p.map = None

    sessions_dir = ROOT / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/sessions", StaticFiles(directory=sessions_dir), name="sessions")
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
    ap.add_argument("--text", help="passage file name in static/texts (without .txt)")
    ap.add_argument("--skim", action="store_true", help="mark this session's reading as deliberately skimmed")
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
    if args.text:
        cfg["reading"]["text"] = args.text
    cfg["reading"]["skim"] = bool(args.skim)
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
        log.warning("%s\nFalling back to interactive synthetic simulator mode (--source synthetic).", e)
        p.source = "synthetic"
        p.open()
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
