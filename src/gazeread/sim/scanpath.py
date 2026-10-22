"""Synthetic reading over a paged layout, with ground truth, plus webcam-like gaze noise (spec 7.3).

Units: px (viewport CSS) and seconds. The model is deliberately simple: fixed-pitch character
grid (no word lengths), left-to-right fixations with regressions, return sweeps, occasional
re-reading of the previous line, and whole paragraphs that are skimmed. Noise per axis is a
slow Ornstein-Uhlenbeck drift (the systematic share measured live) plus white noise, with
blink-like runs of invalid samples, sampled at a fixed frame rate.
"""
from dataclasses import dataclass, field

import numpy as np

from gazeread.types import LineBox


@dataclass
class Layout:
    pitch: float
    font_px: float
    char_w: float
    chars_per_line: int
    pages: list[list[LineBox]]
    feasible: bool                      # fits the viewport (whole paragraphs) with enough characters
    para_lines: list[int] = field(default_factory=list)


def make_layout(pitch: float, cfg: dict, rng: np.random.Generator,
                para_lines: list[int] | None = None) -> Layout:
    """Paragraphs of whole lines on pages of the viewport; a paragraph taller than the page gets
    a virtual taller page (and the layout is marked infeasible)."""
    v, L = cfg["viewport"], cfg["layout"]
    font = pitch / L["line_height"]
    cw = L["char_w_em"] * font
    chars = int(min(L["column_ch"], (v["w"] - 2 * v["margin_x"]) // cw))
    x0 = (v["w"] - chars * cw) / 2
    if para_lines is None:
        lo, hi = L["para_lines"]
        para_lines = [int(rng.integers(lo, hi + 1)) for _ in range(L["n_paras"])]
    usable = v["h"] - 2 * v["margin_y"]
    gap = L["para_gap_lines"] * pitch
    pages, page, y, line_id = [], [], v["margin_y"], 0
    for pid, n in enumerate(para_lines):
        h = n * pitch
        if page and y + gap + h > v["margin_y"] + usable:
            pages.append(page)
            page, y = [], v["margin_y"]
        if page:
            y += gap
        for k in range(n):
            if k == n - 1:
                nch = int(chars * rng.uniform(*L["last_line_frac"]))
            else:
                nch = chars - int(rng.integers(0, L["ragged_chars"] + 1))
            page.append(LineBox(page=len(pages), line_id=line_id, para_id=pid, x0=x0, y0=y,
                                x1=x0 + nch * cw, y1=y + pitch, text="x" * nch))
            line_id += 1
            y += pitch
    pages.append(page)
    feasible = max(para_lines) * pitch <= usable and chars >= L["min_chars_per_line"]
    return Layout(pitch, font, cw, chars, pages, feasible, list(para_lines))


@dataclass
class Reading:
    fix_t0: np.ndarray       # (n,) s
    fix_t1: np.ndarray
    fix_x: np.ndarray        # px
    fix_y: np.ndarray
    fix_line: np.ndarray     # global line_id
    page_t_on: list[float]
    t_end: float


def simulate_reading(layout: Layout, cfg: dict, rng: np.random.Generator) -> Reading:
    R = cfg["reading"]
    cw = layout.char_w

    def duration(mean):
        k = (mean / R["fix_sd_s"]) ** 2
        return float(rng.gamma(k, mean / k))

    t0s, t1s, xs, ys, lines, page_on = [], [], [], [], [], []
    t = 0.0

    def fixate(x, ln, mean):
        nonlocal t
        y = (ln.y0 + ln.y1) / 2 + rng.normal(0, R["fix_y_jitter_pitch"] * layout.pitch)
        d = duration(mean)
        t0s.append(t); t1s.append(t + d); xs.append(x); ys.append(y); lines.append(ln.line_id)
        t += d + R["saccade_s"]

    def read_line(ln, skim):
        sac = R["skim_saccade_chars"] if skim else R["saccade_chars"]
        mean = R["skim_fix_mean_s"] if skim else R["fix_mean_s"]
        x = ln.x0 + R["start_chars"] * cw
        while x <= ln.x1:
            fixate(x, ln, mean)
            if not skim and rng.random() < R["regress_p"]:
                x = max(ln.x0 + cw, x - rng.uniform(*R["regress_chars"]) * cw)
                fixate(x, ln, mean)
            x += rng.uniform(*sac) * cw

    for page in layout.pages:
        t += R["page_turn_s"]
        page_on.append(t - R["page_turn_s"] / 2)
        paras = list(dict.fromkeys(ln.para_id for ln in page))
        prev = None
        for pid in paras:
            skim = rng.random() < R["skim_p"]
            for ln in [l for l in page if l.para_id == pid]:
                if skim and rng.random() < R["skim_line_skip_p"]:
                    continue
                read_line(ln, skim)
                if prev is not None and not skim and rng.random() < R["reread_line_p"]:
                    read_line(prev, False)
                    read_line(ln, False)
                prev = ln
    return Reading(np.array(t0s), np.array(t1s), np.array(xs), np.array(ys), np.array(lines),
                   page_on, t)


def true_dwell(reading: Reading, n_lines: int) -> np.ndarray:
    """Seconds of fixation per global line_id."""
    return np.bincount(reading.fix_line, weights=reading.fix_t1 - reading.fix_t0, minlength=n_lines)


def sample_gaze(reading: Reading, sigma_y: float, x_over_y: float, systematic_frac: float,
                cfg: dict, rng: np.random.Generator):
    """30 fps samples: (t, x, y, valid, true_line) with noise. During saccades the true gaze is
    interpolated between fixations and true_line = -1; during page turns samples are invalid."""
    N = cfg["noise"]
    dt = 1.0 / N["fps"]
    t = np.arange(0.0, reading.t_end, dt)
    k = np.searchsorted(reading.fix_t0, t, side="right") - 1
    k = np.clip(k, 0, len(reading.fix_t0) - 1)
    in_fix = (t >= reading.fix_t0[k]) & (t <= reading.fix_t1[k])
    nxt = np.minimum(k + 1, len(reading.fix_t0) - 1)
    span = np.maximum(reading.fix_t0[nxt] - reading.fix_t1[k], 1e-9)
    a = np.clip((t - reading.fix_t1[k]) / span, 0, 1)
    gx = np.where(in_fix, reading.fix_x[k], (1 - a) * reading.fix_x[k] + a * reading.fix_x[nxt])
    gy = np.where(in_fix, reading.fix_y[k], (1 - a) * reading.fix_y[k] + a * reading.fix_y[nxt])
    true_line = np.where(in_fix & (t >= reading.fix_t0[0]), reading.fix_line[k], -1)
    turning = (t < reading.fix_t0[0]) | ((~in_fix) & (span > 0.3))   # page turn gaps
    valid = ~turning

    sx, sy = x_over_y * sigma_y, sigma_y
    out = []
    for s in (sx, sy):
        sd_d, sd_w = systematic_frac * s, np.sqrt(max(0.0, 1 - systematic_frac ** 2)) * s
        phi = np.exp(-dt / N["drift_tau_s"])
        drift = np.empty(len(t))
        drift[0] = rng.normal(0, sd_d)
        e = rng.normal(0, sd_d * np.sqrt(1 - phi ** 2), len(t))
        for i in range(1, len(t)):
            drift[i] = phi * drift[i - 1] + e[i]
        out.append(drift + rng.normal(0, sd_w, len(t)))
    x, y = gx + out[0], gy + out[1]

    lo, hi = N["blink_frames"]
    n_target, n_bad = int(N["invalid_frac"] * len(t)), 0
    while n_bad < n_target:
        s0, n = int(rng.integers(0, len(t))), int(rng.integers(lo, hi + 1))
        n_bad += int(valid[s0:s0 + n].sum())
        valid[s0:s0 + n] = False
    return t, x, y, valid, true_line
