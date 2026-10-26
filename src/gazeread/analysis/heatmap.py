"""Attention heatmaps and reading intelligence analytics from a session log (spec 6.5).

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


def _evaluate_paragraph(read_s: float, word_count: int, fix_count: int, regressions: int) -> dict:
    """Compute realistic reading speed, engagement status, and plain-English explanation for a paragraph."""
    # Normal human silent reading is ~200-250 WPM (~3.6 words/s).
    expected_s = max(word_count / 3.6, 1.2)
    min_skim_s = max(1.1, 0.22 * expected_s)

    if read_s < min_skim_s or (fix_count == 0 and read_s < 1.5):
        return {
            "status": "Skipped",
            "friendly_status": "⏭️ Skipped Over",
            "wpm": 0.0,
            "est_words_read": int(min(word_count, round(read_s * 3.0))),
            "insight": (
                f"You only looked here for {read_s:.1f}s (a {word_count}-word paragraph normally takes ~{expected_s:.0f}s). "
                "Flagged for review below so you don't miss key points."
            ),
        }

    raw_wpm = (word_count / (read_s / 60.0)) if read_s > 0.3 else 0.0
    reg_ratio = regressions / max(1, fix_count)

    if (read_s > 1.35 * expected_s and regressions >= 2) or reg_ratio >= 0.28 or (read_s > 1.85 * expected_s):
        wpm = min(raw_wpm, 260.0)
        return {
            "status": "Struggled / Re-read",
            "friendly_status": "🔄 Re-read / Tricky",
            "wpm": _r(wpm),
            "est_words_read": int(word_count),
            "insight": (
                f"You spent {read_s:.1f}s here and jumped back to re-read {regressions} time(s). "
                "This section required extra effort — added to your key takeaways below."
            ),
        }

    if raw_wpm > 340.0 or read_s < 0.55 * expected_s:
        # Reader scanned/skimmed rather than reading every single word
        est_read = int(min(word_count, max(4, round(read_s * 3.8))))
        skim_wpm = min(380.0, 240.0 + (raw_wpm - 340.0) * 0.25)
        return {
            "status": "Skimmed",
            "friendly_status": "⚡ Skimmed Quickly",
            "wpm": _r(skim_wpm),
            "est_words_read": est_read,
            "insight": (
                f"You scanned this {word_count}-word paragraph in just {read_s:.1f}s (catching ~{est_read} words). "
                "Good for getting the gist, but details may have been missed."
            ),
        }

    return {
        "status": "Deep Read",
        "friendly_status": "✅ Read Carefully",
        "wpm": _r(min(raw_wpm, 340.0)),
        "est_words_read": int(word_count),
        "insight": (
            f"You read this {word_count}-word section steadily for {read_s:.1f}s (~{round(raw_wpm)} words/min) "
            "with good focus and comprehension flow."
        ),
    }


def _nearest_line_and_para(x: float, y: float, lines: list[LineBox], margin_x: float, margin_y: float) -> tuple[int | None, int | None]:
    if not lines:
        return None, None
    x0 = min(ln.x0 for ln in lines) - margin_x
    x1 = max(ln.x1 for ln in lines) + margin_x
    y0 = min(ln.y0 for ln in lines) - margin_y
    y1 = max(ln.y1 for ln in lines) + margin_y
    if not (x0 <= x <= x1 and y0 <= y <= y1):
        return None, None
    centres = np.array([(ln.y0 + ln.y1) / 2 for ln in lines])
    idx = int(np.argmin(np.abs(centres - y)))
    return int(lines[idx].para_id), int(lines[idx].line_id)


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
    calib = [e for e in events if e["type"] == "calib_result" and e.get("ok") and e.get("primary", True)]
    calib = [e for e in calib if e.get("t", -np.inf) <= t_start] or calib
    if not calib:
        raise ValueError("no successful calibration in this log")
    c = calib[-1]
    first = next(i for i, e in enumerate(events) if e is reading[0])
    meta = [e for e in events[:first] if e["type"] == "meta"][-1]       # in force when the reading began
    gaze = [e for e in events if e["type"] == "gaze" and t_start <= e["t"] <= t_end]
    gt = np.array([g["t"] for g in gaze], float) if gaze else np.zeros(0, float)
    gx = np.array([np.nan if g["x"] is None else g["x"] for g in gaze], float) if gaze else np.zeros(0, float)
    gy = np.array([np.nan if g["y"] is None else g["y"] for g in gaze], float) if gaze else np.zeros(0, float)
    gv = np.array([g["valid"] for g in gaze], bool) if gaze else np.zeros(0, bool)

    sigma, rms_x, rms_y = c["sigma_px"], c["rms_x"], c["rms_y"]
    pitch = reading[0]["line_pitch"]
    thr = cfg["idt_k"] * sigma
    primary_src = cfg["mapping_source"]

    out_pages = []
    timeline = []
    para_agg: dict[int, dict] = {}
    total_fixations = 0
    total_regressions = 0

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

        # Map each fixation to its nearest line/paragraph for regression and return-sweep detection
        fix_records = []
        para_fix_counts: dict[int, int] = {}
        para_regressions: dict[int, int] = {}
        para_sweeps: dict[int, int] = {}
        prev_fix_info = None
        for f in fix:
            pid, lid = _nearest_line_and_para(f.x, f.y, lines, rms_x, 2 * rms_y)
            is_reg = False
            is_sweep = False
            if pid is not None:
                para_fix_counts[pid] = para_fix_counts.get(pid, 0) + 1
                if prev_fix_info is not None and prev_fix_info[0] == pid:
                    prev_pid, prev_lid, prev_x, prev_y = prev_fix_info
                    if (lid == prev_lid and (prev_x - f.x) > max(45.0, 1.2 * rms_x)) or (lid < prev_lid):
                        is_reg = True
                        para_regressions[pid] = para_regressions.get(pid, 0) + 1
                        total_regressions += 1
                    elif lid > prev_lid and (prev_x - f.x) > 120.0:
                        is_sweep = True
                        para_sweeps[pid] = para_sweeps.get(pid, 0) + 1
                prev_fix_info = (pid, lid, f.x, f.y)
            fix_records.append({
                "t0": _r(f.t0), "t1": _r(f.t1),
                "x": _r(f.x), "y": _r(f.y), "dur": _r(f.dur),
                "para_id": pid, "line_id": lid, "regression": is_reg, "return_sweep": is_sweep,
            })
        total_fixations += len(fix)

        # Build per-page sample trace for 1x/2x Animated Gaze Replay
        pt_t, pt_x, pt_y, pt_v = gt[m], gx[m], gy[m], gv[m]
        samples_trace = []
        step = max(1, len(pt_t) // 600)
        for idx in range(0, len(pt_t), step):
            sv = bool(pt_v[idx] and np.isfinite(pt_x[idx]) and np.isfinite(pt_y[idx]))
            samples_trace.append({
                "t": _r(pt_t[idx] - t_on),
                "t_session": _r(pt_t[idx] - t_start),
                "x": _r(pt_x[idx]) if sv else None,
                "y": _r(pt_y[idx]) if sv else None,
                "valid": sv,
            })
            if sv and (idx % max(1, step * 3) == 0):
                pid, lid = _nearest_line_and_para(float(pt_x[idx]), float(pt_y[idx]), lines, rms_x, 2 * rms_y)
                timeline.append({
                    "t": _r(pt_t[idx] - t_start),
                    "page": int(pe["page"]),
                    "para_id": pid,
                    "line_id": lid,
                    "x": _r(pt_x[idx]),
                    "y": _r(pt_y[idx]),
                })

        # Compute per-paragraph analytics on this page using continuous sample dwell for realistic time
        prim_dwell = dwell[primary_src]
        samp_dwell = dwell["samples"]
        samp_map = dict(zip(samp_dwell["para_ids"], samp_dwell["para_s"]))
        para_metrics = []
        for pid, ps, pf in zip(prim_dwell["para_ids"], prim_dwell["para_s"], prim_dwell["para_frac"]):
            p_lines = [ln for ln in lines if ln.para_id == pid]
            full_text = " ".join(ln.text for ln in p_lines).strip()
            words = len(full_text.split())
            fc = para_fix_counts.get(pid, 0)
            regs = para_regressions.get(pid, 0)
            sweeps = para_sweeps.get(pid, 0)
            # Use max of sample dwell and fixation dwell so inter-word saccades aren't lost
            read_s = max(float(samp_map.get(pid, ps)), float(ps))
            eval_info = _evaluate_paragraph(read_s, words, fc, regs)
            pm = {
                "para_id": int(pid),
                "page": int(pe["page"]),
                "word_count": int(words),
                "est_words_read": eval_info["est_words_read"],
                "line_count": len(p_lines),
                "dwell_s": _r(read_s),
                "dwell_frac": _r(pf),
                "wpm": eval_info["wpm"],
                "fixation_count": int(fc),
                "regressions": int(regs),
                "return_sweeps": int(sweeps),
                "status": eval_info["status"],
                "friendly_status": eval_info["friendly_status"],
                "insight": eval_info["insight"],
                "text": full_text,
            }
            para_metrics.append(pm)
            if pid not in para_agg:
                para_agg[pid] = dict(pm)
            else:
                agg = para_agg[pid]
                agg["word_count"] += int(words)
                agg["line_count"] += len(p_lines)
                agg["dwell_s"] = _r(agg["dwell_s"] + read_s)
                agg["fixation_count"] += int(fc)
                agg["regressions"] += int(regs)
                agg["return_sweeps"] += int(sweeps)
                agg["text"] = (agg["text"] + " " + full_text).strip()

        out_pages.append({
            "page": pe["page"], "t_on": _r(t_on), "t_off": _r(t_off),
            "lines": pe["lines"],
            "fixations": [{"t0": f["t0"], "t1": f["t1"], "x": f["x"], "y": f["y"], "dur": f["dur"]}
                          for f in fix_records],
            "fixation_details": fix_records,
            "samples_trace": samples_trace,
            "para_metrics": para_metrics,
            "n_gaze": int(m.sum()), "n_gaze_valid": int((m & gv).sum()),
            "dwell": dwell,
        })

    # Finalize passage-wide paragraph summary
    total_dwell_s = sum(p["dwell_s"] for p in para_agg.values())
    total_words = sum(p["word_count"] for p in para_agg.values())
    reading_dur_s = max(0.001, float(t_end - t_start))
    summary_paras = []
    for pid in sorted(para_agg):
        p = para_agg[pid]
        ds = p["dwell_s"]
        wc = p["word_count"]
        p["dwell_frac"] = _r(ds / total_dwell_s) if total_dwell_s > 0 else 0.0
        eval_info = _evaluate_paragraph(ds, wc, p["fixation_count"], p["regressions"])
        p["wpm"] = eval_info["wpm"]
        p["est_words_read"] = eval_info["est_words_read"]
        p["status"] = eval_info["status"]
        p["friendly_status"] = eval_info["friendly_status"]
        p["insight"] = eval_info["insight"]
        summary_paras.append(p)

    # Compute realistic overall WPM based on estimated words actually read vs active reading time
    total_est_words_read = sum(p["est_words_read"] for p in summary_paras)
    read_paras = [p for p in summary_paras if p["status"] in ("Deep Read", "Struggled / Re-read")]
    if read_paras:
        read_w = sum(p["word_count"] for p in read_paras)
        read_t = max(1.0, sum(p["dwell_s"] for p in read_paras))
        overall_wpm = round(min(350.0, (read_w / (read_t / 60.0))), 1)
    elif total_est_words_read > 0 and reading_dur_s > 0.5:
        overall_wpm = round(min(360.0, (total_est_words_read / (reading_dur_s / 60.0))), 1)
    else:
        overall_wpm = 0.0

    valid_ratio = float(gv.sum() / len(gv)) if len(gv) else 0.0
    reg_rate = total_regressions / max(1, total_fixations)
    focus_score = round(min(100.0, max(0.0, 100.0 * (0.65 * valid_ratio + 0.35 * max(0.0, 1.0 - reg_rate)))), 1)

    n_deep = sum(1 for p in summary_paras if p["status"] == "Deep Read")
    n_tricky = sum(1 for p in summary_paras if p["status"] == "Struggled / Re-read")
    n_skim = sum(1 for p in summary_paras if p["status"] == "Skimmed")
    n_skip = sum(1 for p in summary_paras if p["status"] == "Skipped")
    n_tot_p = len(summary_paras)

    if n_deep + n_tricky == n_tot_p and n_tricky == 0:
        verdict_headline = (
            f"You read all {n_tot_p} paragraphs carefully at a comfortable ~{round(overall_wpm)} words/min "
            f"with strong focus ({round(focus_score)}%)."
        )
    elif n_skip + n_skim >= max(1, n_tot_p // 2):
        verdict_headline = (
            f"You moved quickly through the passage in {reading_dur_s:.0f}s — reading {n_deep + n_tricky} of {n_tot_p} "
            f"paragraphs carefully and skimming/skipping {n_skip + n_skim}. Check the review cards below for what you missed!"
        )
    elif n_tricky > 0:
        tricky_ids = ", ".join(f"#{p['para_id'] + 1}" for p in summary_paras if p["status"] == "Struggled / Re-read")
        verdict_headline = (
            f"You read most of the passage steadily (~{round(overall_wpm)} words/min), but found Paragraph {tricky_ids} "
            f"tricky and re-read {total_regressions} times."
        )
    else:
        verdict_headline = (
            f"You read ~{total_est_words_read} of {total_words} words in {reading_dur_s:.0f}s "
            f"(~{round(overall_wpm)} words/min) with {round(focus_score)}% focus."
        )

    gran = choose_granularity(rms_y, pitch)
    summary = {
        "reading_duration_s": _r(reading_dur_s),
        "total_attended_s": _r(total_dwell_s),
        "total_words": int(total_words),
        "est_words_read": int(total_est_words_read),
        "overall_wpm": _r(overall_wpm),
        "total_fixations": int(total_fixations),
        "total_regressions": int(total_regressions),
        "regression_rate": _r(reg_rate),
        "valid_gaze_ratio": _r(valid_ratio),
        "focus_score": _r(focus_score),
        "verdict_headline": verdict_headline,
        "counts": {"deep": n_deep, "tricky": n_tricky, "skimmed": n_skim, "skipped": n_skip},
        "paragraphs": summary_paras,
    }

    return {
        "version": RESULT_VERSION, "source": meta.get("source"),
        "viewport": [meta["screen_w"], meta["screen_h"]],
        "sigma_px": _r(sigma), "rms_x": _r(rms_x), "rms_y": _r(rms_y),
        "line_pitch": pitch, "font_px": reading[0]["font_px"], "granularity": gran,
        "mapping_mode": cfg["mapping_mode"], "primary_source": cfg["mapping_source"],
        "idt_threshold_px": _r(thr), "pages": out_pages,
        "timeline": timeline, "summary": summary,
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
