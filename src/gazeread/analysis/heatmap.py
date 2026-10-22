"""Attention heatmaps from a session log (spec 6.5).

`analyze_session` is a pure function of the logged events: the live app and
scripts/replay_session.py both call it on events.jsonl, so replay reproduces the live result.
It analyses the most recent reading (the last `page` event with page == 0 up to `reading_end`).
Both mapping sources are always computed: "fixations" (I-DT) and "samples" (every gaze sample).
"""
import json
from pathlib import Path

import numpy as np

from gazeread.analysis.fixations import idt, merge
from gazeread.analysis.line_mapping import line_dwell, normalize, para_dwell, sample_points
from gazeread.types import LineBox

RESULT_VERSION = 1


def choose_granularity(rms_y: float, line_pitch: float) -> str:
    """'line' only if the vertical error is at most half the line pitch, else 'paragraph'."""
    return "line" if rms_y <= line_pitch / 2 else "paragraph"


def _r(v: float) -> float:
    return round(float(v), 4)


def _dwell_block(line_s: np.ndarray, lines: list[LineBox]) -> dict:
    ids, para_s = para_dwell(line_s, lines)
    return {"line_s": [_r(v) for v in line_s], "line_frac": [_r(v) for v in normalize(line_s)],
            "para_ids": ids, "para_s": [_r(v) for v in para_s],
            "para_frac": [_r(v) for v in normalize(para_s)]}


def analyze_session(events: list[dict], cfg: dict) -> dict:
    """Heatmap result for the latest reading in `events`. `cfg` is the `analysis` config block."""
    pages = [e for e in events if e["type"] == "page"]
    ends = [e for e in events if e["type"] == "reading_end"]
    starts = [i for i, e in enumerate(pages) if e["page"] == 0]
    if not starts or not ends:
        raise ValueError("no complete reading (page 0 ... reading_end) in this log")
    reading = pages[starts[-1]:]
    t_end = next(e["t"] for e in reversed(ends))
    t_start = reading[0]["t_on"]
    calib = [e for e in events if e["type"] == "calib_result" and e.get("ok")]
    calib = [e for e in calib if e.get("t", -np.inf) <= t_start] or calib
    if not calib:
        raise ValueError("no successful calibration in this log")
    c = calib[-1]
    meta = [e for e in events if e["type"] == "meta"][-1]
    gaze = [e for e in events if e["type"] == "gaze" and t_start <= e["t"] <= t_end]
    gt = np.array([g["t"] for g in gaze])
    gx = np.array([np.nan if g["x"] is None else g["x"] for g in gaze])
    gy = np.array([np.nan if g["y"] is None else g["y"] for g in gaze])
    gv = np.array([g["valid"] for g in gaze], bool)

    sigma, rms_x, rms_y = c["sigma_px"], c["rms_x"], c["rms_y"]
    pitch = reading[0]["line_pitch"]
    thr = cfg["idt_k"] * sigma
    out_pages = []
    for k, pe in enumerate(reading):
        t_on = pe["t_on"]
        t_off = reading[k + 1]["t_on"] if k + 1 < len(reading) else t_end
        lines = [LineBox(page=pe["page"], **ln) for ln in pe["lines"]]
        m = (gt >= t_on) & (gt < t_off)
        fix = merge(idt(gt[m], gx[m], gy[m], gv[m], thr, cfg["min_fix_s"]), cfg["merge_gap_s"], thr)
        sources = {
            "fixations": [(f.x, f.y, f.dur) for f in fix],
            "samples": sample_points(gt[m], gx[m], gy[m], gv[m], cfg["max_sample_dt_s"]),
        }
        dwell = {name: _dwell_block(line_dwell(pts, lines, cfg["mapping_mode"], rms_y,
                                               cfg["hard_tol_lines"] * pitch, rms_x, 2 * rms_y), lines)
                 for name, pts in sources.items()}
        out_pages.append({
            "page": pe["page"], "t_on": _r(t_on), "t_off": _r(t_off),
            "lines": pe["lines"],
            "fixations": [{"t0": _r(f.t0), "t1": _r(f.t1), "x": _r(f.x), "y": _r(f.y), "dur": _r(f.dur)}
                          for f in fix],
            "n_gaze": int(m.sum()), "n_gaze_valid": int((m & gv).sum()),
            "dwell": dwell,
        })
    return {
        "version": RESULT_VERSION, "source": meta.get("source"),
        "viewport": [meta["screen_w"], meta["screen_h"]],
        "sigma_px": _r(sigma), "rms_x": _r(rms_x), "rms_y": _r(rms_y),
        "line_pitch": pitch, "font_px": reading[0]["font_px"], "granularity": choose_granularity(rms_y, pitch),
        "mapping_mode": cfg["mapping_mode"], "primary_source": cfg["mapping_source"],
        "idt_threshold_px": _r(thr), "pages": out_pages,
    }


def dumps(result: dict) -> str:
    """Canonical JSON text (used for the saved file and the replay comparison)."""
    return json.dumps(result, sort_keys=True, indent=1, ensure_ascii=False)


def render_png(result: dict, page_idx: int, path: str | Path, view: str = "heatmap") -> None:
    """Draw one page: view 'heatmap' (line + paragraph dwell) or 'scanpath' (coarse)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    w, h = result["viewport"]
    page = result["pages"][page_idx]
    d = page["dwell"][result["primary_source"]]
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    cmap = plt.get_cmap("YlOrRd")
    para_frac = dict(zip(d["para_ids"], d["para_frac"]))
    pmax = max(para_frac.values()) or 1.0
    lmax = max(d["line_frac"]) or 1.0
    lines = page["lines"]
    if view == "heatmap":
        for pid in d["para_ids"]:
            pl = [ln for ln in lines if ln["para_id"] == pid]
            x0, y0 = min(l["x0"] for l in pl), min(l["y0"] for l in pl)
            x1, y1 = max(l["x1"] for l in pl), max(l["y1"] for l in pl)
            alpha = 0.55 if result["granularity"] == "paragraph" else 0.15
            ax.add_patch(Rectangle((x0 - 24, y0), 14, y1 - y0, color=cmap(para_frac[pid] / pmax)))
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color=cmap(para_frac[pid] / pmax), alpha=alpha))
        if result["granularity"] == "line":
            for ln, f in zip(lines, d["line_frac"]):
                ax.add_patch(Rectangle((ln["x0"], ln["y0"]), ln["x1"] - ln["x0"], ln["y1"] - ln["y0"],
                                       color=cmap(f / lmax), alpha=0.5))
    for ln in lines:
        ax.text(ln["x0"], (ln["y0"] + ln["y1"]) / 2, ln["text"], va="center", ha="left",
                fontsize=0.72 * result["font_px"], family=["Georgia", "Times New Roman", "DejaVu Serif"], color="#222")
    if view == "scanpath" and page["fixations"]:
        fx = [f["x"] for f in page["fixations"]]
        fy = [f["y"] for f in page["fixations"]]
        ax.plot(fx, fy, "-", color="#1f5fbf", lw=1, alpha=0.6)
        ax.scatter(fx, fy, s=[40 + 400 * f["dur"] for f in page["fixations"]], color="#1f5fbf", alpha=0.45)
    gran = result["granularity"]
    label = {"heatmap": f"attention heatmap: {gran} level"
             + ("" if gran == "line" else " (line-level low confidence)"),
             "scanpath": "scanpath (coarse)"}[view]
    label += (f" | page {page['page'] + 1} | sigma_y {result['rms_y']:.0f} px, line pitch "
              f"{result['line_pitch']:.0f} px | source {result['primary_source']}")
    if result["source"] == "synthetic":
        label = "SYNTHETIC (not a measurement) | " + label
    ax.text(12, h - 14, label, fontsize=11, color="#333")
    fig.savefig(path)
    plt.close(fig)


def export(result: dict, out_dir: str | Path, json_name: str = "heatmap.json") -> list[Path]:
    """Write the result JSON plus heatmap/scanpath PNGs per page. Returns the written paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / json_name]
    paths[0].write_text(dumps(result), encoding="utf-8")
    for i in range(len(result["pages"])):
        for view in ("heatmap", "scanpath"):
            p = out_dir / f"{view}_p{i + 1}.png"
            render_png(result, i, p, view)
            paths.append(p)
    return paths
