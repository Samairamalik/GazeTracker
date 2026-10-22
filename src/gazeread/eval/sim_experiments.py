"""M4 simulation experiments (spec 7.3): heatmap fidelity vs noise, line pitch and paragraph height.

    python -m gazeread.eval.sim_experiments [--seeds N]     # `make sim`

Every simulated reading is turned into a session-log-shaped event list and analysed with the
app's own `analyze_session` (same I-DT, merge, line mapping, granularity code and the app's
config), so these numbers describe the shipped pipeline. Noise level and shape come from
results/tables/measured_sigma_summary.csv (live calibrations). Outputs:
results/tables/sim_results.csv, sim_guideline.csv, sim_para_height.csv,
results/figures/sim_*.png and results/sim_summary.md. Fixed seeds; reruns are identical.
"""
import argparse
import csv
import math
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from gazeread.analysis.heatmap import analyze_session
from gazeread.eval.metrics import assignment_accuracy, bootstrap_ci, spearman
from gazeread.gaze.filters import make_filter
from gazeread.sim.scanpath import make_layout, sample_gaze, simulate_reading, true_dwell
from gazeread.types import LineBox

ROOT = Path(__file__).resolve().parents[3]
TABLES, FIGS = ROOT / "results" / "tables", ROOT / "results" / "figures"
METRICS = ("rho_line", "rho_line_within", "rho_para", "acc_line", "acc_para", "fix_time_frac")


def load_cfg() -> dict:
    sim = yaml.safe_load((ROOT / "configs/sim.yaml").read_text())
    app = yaml.safe_load((ROOT / "configs/default.yaml").read_text())
    sim["analysis"], sim["filter"] = app["analysis"], app["filter"]
    with (TABLES / "measured_sigma_summary.csv").open() as f:
        sim["measured"] = {r["key"]: float(r["value"]) for r in csv.DictReader(f)}
    return sim


def within_para_rho(truth: np.ndarray, est: np.ndarray, paras: np.ndarray, min_lines: int = 3) -> float:
    """Mean Spearman between true and estimated line dwell computed *within* each paragraph
    (paragraphs with >= min_lines lines and non-constant values). Unlike the pooled line rho,
    this is not lifted by between-paragraph contrast, so it measures line resolution itself."""
    vals = []
    for p in np.unique(paras):
        m = paras == p
        if m.sum() >= min_lines:
            r = spearman(truth[m], est[m])
            if np.isfinite(r):
                vals.append(r)
    return float(np.mean(vals)) if vals else float("nan")


def run_one(task: dict) -> list[dict]:
    """One simulated reading at one noise level, analysed under every filter x mapping x source."""
    cfg, s = task["cfg"], task["seed"]
    m = cfg["measured"]
    lay_rng = np.random.default_rng([cfg["seed"], s, 1])           # same text/reading across sigma
    layout = make_layout(task["pitch"], cfg, lay_rng, task.get("para_lines"))
    reading = simulate_reading(layout, cfg, lay_rng)
    noise_rng = np.random.default_rng([cfg["seed"], s, 2, int(task["sigma_y"] * 10)])
    t, x, y, valid, true_line = sample_gaze(reading, task["sigma_y"], m["x_over_y"],
                                            m["systematic_frac"], cfg, noise_rng)
    lines_all = [ln for pg in layout.pages for ln in pg]
    truth_line = true_dwell(reading, len(lines_all))
    paras = np.array([ln.para_id for ln in lines_all])
    truth_para = np.bincount(paras, weights=truth_line)
    sx, sy = m["x_over_y"] * task["sigma_y"], task["sigma_y"]

    rows = []
    true_fix_s = float((reading.fix_t1 - reading.fix_t0).sum())
    for use_filter in task.get("filters", cfg["grid"]["filter"]):
        if use_filter:
            f = make_filter(cfg["filter"])
            g = [f(ti, xi, yi, vi) for ti, xi, yi, vi in zip(t, x, y, valid)]
            gx, gy, gv = (np.array([p.x for p in g]), np.array([p.y for p in g]),
                          np.array([p.valid for p in g]))
        else:
            gx, gy, gv = np.where(valid, x, np.nan), np.where(valid, y, np.nan), valid
        events = [{"type": "meta", "screen_w": cfg["viewport"]["w"], "screen_h": cfg["viewport"]["h"],
                   "source": "simulation"},
                  {"type": "calib_result", "ok": True, "primary": True, "t": -1.0,
                   "sigma_px": math.hypot(sx, sy), "rms_x": sx, "rms_y": sy}]
        for k, pg in enumerate(layout.pages):
            events.append({"type": "page", "page": k, "t_on": reading.page_t_on[k],
                           "lines": [{kk: getattr(ln, kk) for kk in ("line_id", "para_id", "x0", "y0", "x1", "y1", "text")}
                                     for ln in pg],
                           "line_pitch": layout.pitch, "font_px": layout.font_px, "n_pages": len(layout.pages)})
        events += [{"type": "gaze", "t": float(ti), "x": None if not vi else float(xi),
                    "y": None if not vi else float(yi), "valid": bool(vi)}
                   for ti, xi, yi, vi in zip(t, gx, gy, gv)]
        events.append({"type": "reading_end", "t": reading.t_end})
        for mode in task.get("mappings", cfg["grid"]["mapping"]):
            acfg = dict(cfg["analysis"], mapping_mode=mode, idt_k=task.get("idt_k", cfg["analysis"]["idt_k"]))
            res = analyze_session(events, acfg)
            for source in ("fixations", "samples"):
                est_line = np.concatenate([pg["dwell"][source]["line_s"] for pg in res["pages"]])
                est_para = np.bincount(paras, weights=est_line)
                ok_l = ok_p = cred = 0.0
                for k, pg in enumerate(res["pages"]):
                    lines = layout.pages[k]
                    t0, t1 = pg["t_on"], pg["t_off"]
                    if source == "fixations":
                        pts = []
                        for fx in pg["fixations"]:
                            i = min(np.searchsorted(t, (fx["t0"] + fx["t1"]) / 2), len(t) - 1)
                            if true_line[i] >= 0:
                                pts.append((fx["x"], fx["y"], fx["dur"], int(true_line[i])))
                    else:
                        sel = (t >= t0) & (t < t1) & gv & np.isfinite(gx) & (true_line >= 0)
                        idx = np.flatnonzero(sel)
                        dt = np.minimum(np.diff(t, append=t[-1])[idx], cfg["analysis"]["max_sample_dt_s"])
                        pts = list(zip(gx[idx], gy[idx], dt, true_line[idx]))
                    a, b, c = assignment_accuracy(pts, lines, mode, sy, cfg["analysis"]["hard_tol_lines"] * layout.pitch,
                                                  sx, 2 * sy)
                    ok_l, ok_p, cred = ok_l + a, ok_p + b, cred + c
                rows.append({
                    "experiment": task["experiment"], "sigma_label": task["sigma_label"],
                    "idt_k": acfg["idt_k"],
                    "sigma_y": task["sigma_y"], "pitch": task["pitch"],
                    "para_lines": task.get("para_lines_label", "4-6"), "feasible": int(layout.feasible),
                    "chars_per_line": layout.chars_per_line, "seed": s, "filter": int(use_filter),
                    "mapping": mode, "source": source,
                    "rho_line": spearman(truth_line, est_line),
                    "rho_line_within": within_para_rho(truth_line, est_line, paras),
                    "rho_para": spearman(truth_para, est_para),
                    "acc_line": ok_l / cred if cred else float("nan"),
                    "acc_para": ok_p / cred if cred else float("nan"),
                    "fix_time_frac": (sum(f["dur"] for pg in res["pages"] for f in pg["fixations"]) / true_fix_s
                                      if source == "fixations" else float("nan")),
                })
    return rows


def tasks(cfg: dict, n_seeds: int) -> list[dict]:
    g, m = cfg["grid"], cfg["measured"]
    measured = [("A median", m["rms_y_A_median"]), ("A best", m["rms_y_A_best"])]
    if math.isfinite(m["rms_y_B_median"]):
        measured.append(("B median", m["rms_y_B_median"]))
    out = []
    for s in range(n_seeds):
        for sy in g["sigma_y"]:
            for p in g["pitch"]:
                out.append({"experiment": "grid", "sigma_label": f"{sy:g}", "sigma_y": sy, "pitch": p, "seed": s})
        for label, sy in measured:
            for p in g["pitch"]:
                out.append({"experiment": "guideline", "sigma_label": label, "sigma_y": sy, "pitch": p, "seed": s})
        for label, sy in measured + [(f"{v:g}", v) for v in g["sigma_y"]]:
            for L in g["para_lines_sweep"]:
                n = g["para_sweep_n_paras"]          # same number of paragraphs at every height
                out.append({"experiment": "para", "sigma_label": label, "sigma_y": sy, "pitch": g["base_pitch"],
                            "seed": s, "para_lines": [L] * n, "para_lines_label": str(L)})
        for sy in g["sigma_y"]:                        # I-DT threshold sweep (fixations only matter)
            for k in g["idt_k"]:
                out.append({"experiment": "idt", "sigma_label": f"{sy:g}", "sigma_y": sy, "pitch": g["base_pitch"],
                            "seed": s, "idt_k": k, "filters": [True], "mappings": ["soft"]})
    for t in out:
        t["cfg"] = cfg
    return out


def aggregate(rows: list[dict]) -> list[dict]:
    keys = ("experiment", "sigma_label", "sigma_y", "pitch", "para_lines", "idt_k", "filter", "mapping", "source")
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    out = []
    for key, rs in sorted(groups.items(), key=lambda kv: tuple(str(v) for v in kv[0])):
        row = dict(zip(keys, key))
        row["feasible"] = int(all(r["feasible"] for r in rs))
        row["chars_per_line"] = rs[0]["chars_per_line"]
        row["n"] = len(rs)
        for m_ in METRICS:
            mean, lo, hi = bootstrap_ci([r[m_] for r in rs])
            row[m_], row[m_ + "_lo"], row[m_ + "_hi"] = (round(v, 4) for v in (mean, lo, hi))
        out.append(row)
    return out


def threshold(xs, ys, target: float, feasible, whole: bool = False) -> str:
    """Smallest x (linear interpolation between consecutive feasible grid points) with y >= target.
    whole=True rounds up to an integer (e.g. lines per paragraph, which cannot be fractional)."""
    pts = [(x, y) for x, y, f in zip(xs, ys, feasible) if f and math.isfinite(y)]
    for i, (x, y) in enumerate(pts):
        if y >= target:
            if i == 0:
                return f"<= {x:g}"
            x0, y0 = pts[i - 1]
            xt = x0 + (target - y0) * (x - x0) / (y - y0)
            return f"{math.ceil(xt - 1e-9)}" if whole else f"{xt:.0f}"
    return f"not reached (feasible <= {pts[-1][0]:g})" if pts else "no feasible layout"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def sel(agg, **kw):
    return [r for r in agg if all(r[k] == v for k, v in kw.items())]


PRIMARY = {"filter": 1, "mapping": "soft", "source": "fixations"}     # the app's defaults
PRIMARY_SAMPLES = {"filter": 1, "mapping": "soft", "source": "samples"}  # guideline source (I-DT finding)


def guideline_tables(agg: list[dict], cfg: dict) -> tuple[list[dict], list[dict]]:
    target = cfg["target_rho"]
    gl, ph = [], []
    for label in dict.fromkeys(r["sigma_label"] for r in sel(agg, experiment="guideline")):
        for source in ("fixations", "samples"):
            rs = sorted(sel(agg, experiment="guideline", sigma_label=label, filter=1, mapping="soft", source=source),
                        key=lambda r: r["pitch"])
            gl.append({"sigma_label": label, "sigma_y": round(rs[0]["sigma_y"], 1), "source": source,
                       "min_pitch_within_rho_0.7": threshold([r["pitch"] for r in rs], [r["rho_line_within"] for r in rs],
                                                             target, [r["feasible"] for r in rs]),
                       "min_pitch_pooled_rho_0.7": threshold([r["pitch"] for r in rs], [r["rho_line"] for r in rs],
                                                             target, [r["feasible"] for r in rs]),
                       **{f"within_rho@{r['pitch']:g}{'' if r['feasible'] else '*'}": r["rho_line_within"] for r in rs},
                       "acc_line@66": next(r["acc_line"] for r in rs if r["pitch"] == cfg["grid"]["base_pitch"]),
                       "rho_para@66": next(r["rho_para"] for r in rs if r["pitch"] == cfg["grid"]["base_pitch"])})
        rs = sorted(sel(agg, experiment="para", sigma_label=label, **PRIMARY_SAMPLES), key=lambda r: int(r["para_lines"]))
        ph.append({"sigma_label": label, "sigma_y": round(rs[0]["sigma_y"], 1),
                   "min_para_lines_rho_0.7": threshold([int(r["para_lines"]) for r in rs], [r["rho_para"] for r in rs],
                                                       target, [r["feasible"] for r in rs], whole=True),
                   **{f"rho_para@{r['para_lines']}L{'' if r['feasible'] else '*'}": r["rho_para"] for r in rs}})
    return gl, ph


def figures(agg: list[dict], cfg: dict) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid_c = "#1f1f1e", "#6b6a64", "#e6e4dd"
    cat = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]           # categorical, fixed order
    ramp = ["#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#0d366b"]   # ordinal blue
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": grid_c, "grid.linewidth": 0.8, "lines.linewidth": 2})
    m = cfg["measured"]
    marks = [("A median", m["rms_y_A_median"]), ("A best", m["rms_y_A_best"])]
    target, base = cfg["target_rho"], cfg["grid"]["base_pitch"]
    labels = list(dict.fromkeys(r["sigma_label"] for r in sel(agg, experiment="guideline")))
    paths = []

    def save(fig, name):
        fig.tight_layout()
        path = FIGS / name
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    def style(ax, title, ylabel, xlabel, ylim=(-0.2, 1.05)):
        ax.set_title(title, loc="left", color=ink, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(frameon=False, fontsize=8.5, labelcolor=ink)

    def mark_sigma(ax):
        for lbl, v in marks:
            ax.axvline(v, color=muted, lw=1, ls=":")
            ax.annotate(f"measured {lbl}", (v, 1.0), xytext=(3, 0), textcoords="offset points",
                        color=muted, fontsize=8, rotation=90, va="top")

    # 1. within-paragraph line rho vs sigma, one line per pitch (dashed = does not fit the page)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for c, p in zip(ramp, cfg["grid"]["pitch"]):
        rs = sorted(sel(agg, experiment="grid", pitch=p, **PRIMARY_SAMPLES), key=lambda r: r["sigma_y"])
        ax.plot([r["sigma_y"] for r in rs], [r["rho_line_within"] for r in rs], "-" if rs[0]["feasible"] else "--",
                color=c, marker="o", ms=4, label=f"pitch {p:g} px" + ("" if rs[0]["feasible"] else " (doesn't fit)"))
    ax.axhline(target, color=ink, lw=1)
    mark_sigma(ax)
    style(ax, "Line resolution vs vertical gaze error (samples)", "within-paragraph line ρ", "vertical gaze error σ_y (px)")
    save(fig, "sim_rho_line_vs_sigma.png")

    # 2. within-paragraph line rho vs pitch at the measured sigma values
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for c, lbl in zip(cat, labels):
        rs = sorted(sel(agg, experiment="guideline", sigma_label=lbl, **PRIMARY_SAMPLES), key=lambda r: r["pitch"])
        xs = [r["pitch"] for r in rs]
        ax.plot(xs, [r["rho_line_within"] for r in rs], color=c, marker="o", ms=4,
                label=f"{lbl} (σ_y {rs[0]['sigma_y']:.0f} px)")
        ax.fill_between(xs, [r["rho_line_within_lo"] for r in rs], [r["rho_line_within_hi"] for r in rs],
                        color=c, alpha=0.15, lw=0)
    infeas = [r["pitch"] for r in sel(agg, experiment="guideline", sigma_label=labels[0], **PRIMARY_SAMPLES) if not r["feasible"]]
    if infeas:
        ax.axvspan(min(infeas) - 5, max(infeas) + 5, color=grid_c, alpha=0.7, lw=0)
        ax.annotate("paragraphs don't fit 1512×945", (min(infeas), -0.15), color=muted, fontsize=8)
    ax.axhline(target, color=ink, lw=1)
    style(ax, "Line resolution vs line pitch at our measured error", "within-paragraph line ρ (mean, 95% CI)",
          "line pitch (px)")
    save(fig, "sim_rho_line_vs_pitch.png")

    # 3. line-assignment accuracy vs sigma
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for c, p in zip(ramp, cfg["grid"]["pitch"]):
        rs = sorted(sel(agg, experiment="grid", pitch=p, **PRIMARY_SAMPLES), key=lambda r: r["sigma_y"])
        ax.plot([r["sigma_y"] for r in rs], [r["acc_line"] for r in rs], "-" if rs[0]["feasible"] else "--",
                color=c, marker="o", ms=4, label=f"pitch {p:g} px")
    mark_sigma(ax)
    style(ax, "Share of credited time on the correct line (samples)", "line-assignment accuracy",
          "vertical gaze error σ_y (px)", (0, 1.05))
    save(fig, "sim_line_accuracy_vs_sigma.png")

    # 4. ablations at the base pitch (fixations pipeline as shipped, plus the samples source)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    variants = [("mapping", ["soft", "hard"], "mapping"), ("filter", [1, 0], "One Euro filter"),
                ("source", ["fixations", "samples"], "time source")]
    for ax, (key, vals, title) in zip(axes, variants):
        for c, v in zip(cat, vals):
            rs = sorted(sel(agg, experiment="grid", pitch=base, **dict(PRIMARY, **{key: v})), key=lambda r: r["sigma_y"])
            name = {1: "on", 0: "off"}.get(v, v) if key == "filter" else v
            ax.plot([r["sigma_y"] for r in rs], [r["rho_line_within"] for r in rs], color=c, marker="o", ms=4, label=str(name))
        ax.axhline(target, color=ink, lw=1)
        style(ax, title, f"within-paragraph line ρ, pitch {base:g} px" if ax is axes[0] else "", "σ_y (px)")
    save(fig, "sim_ablations.png")

    # 5. paragraph rho vs paragraph height at the measured sigma values
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for c, lbl in zip(cat, labels):
        rs = sorted(sel(agg, experiment="para", sigma_label=lbl, **PRIMARY_SAMPLES), key=lambda r: int(r["para_lines"]))
        xs = [int(r["para_lines"]) for r in rs]
        ax.plot(xs, [r["rho_para"] for r in rs], color=c, marker="o", ms=4, label=f"{lbl} (σ_y {rs[0]['sigma_y']:.0f} px)")
        ax.fill_between(xs, [r["rho_para_lo"] for r in rs], [r["rho_para_hi"] for r in rs], color=c, alpha=0.15, lw=0)
    ax.axhline(target, color=ink, lw=1)
    style(ax, f"Paragraph resolution vs paragraph height (pitch {base:g} px, 6 paragraphs)",
          "paragraph ρ (mean, 95% CI)", "lines per paragraph")
    save(fig, "sim_rho_para_vs_para_height.png")

    # 6. I-DT threshold sweep: share of fixation time recovered, and line resolution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for c, k in zip(ramp[1::2], cfg["grid"]["idt_k"]):
        rs = sorted(sel(agg, experiment="idt", idt_k=k, filter=1, mapping="soft", source="fixations"),
                    key=lambda r: r["sigma_y"])
        xs = [r["sigma_y"] for r in rs]
        axes[0].plot(xs, [r["fix_time_frac"] for r in rs], color=c, marker="o", ms=4, label=f"idt_k {k:g}")
        axes[1].plot(xs, [r["rho_line_within"] for r in rs], color=c, marker="o", ms=4, label=f"idt_k {k:g}")
    style(axes[0], "Fixation time recovered by I-DT", "detected / true fixation time", "σ_y (px)", (0, 1.5))
    axes[1].axhline(target, color=ink, lw=1)
    style(axes[1], f"Line resolution with I-DT fixations (pitch {base:g} px)", "within-paragraph line ρ", "σ_y (px)")
    save(fig, "sim_idt_sweep.png")
    return paths


def summary_md(agg, gl, ph, idt_rows, cfg, runtime_s, n_runs) -> str:
    m = cfg["measured"]
    base = cfg["grid"]["base_pitch"]
    L = ["# M4 simulation summary (generated by `make sim`; do not edit by hand)", "",
         f"- {n_runs} simulated readings, {cfg['n_seeds']} seeds per condition, runtime {runtime_s:.0f} s.",
         f"- Noise shape from {int(m['n_included'])} live calibrations: σ_x/σ_y = {m['x_over_y']:.2f}, "
         f"systematic share {m['systematic_frac']:.2f} (OU drift, τ {cfg['noise']['drift_tau_s']} s).",
         "- Line metric: **within-paragraph ρ** (Spearman of true vs estimated line dwell inside each paragraph, "
         "averaged). The spec's pooled ρ over all lines is also given; it is lifted by between-paragraph contrast.",
         f"- Pipeline: One Euro on, soft mapping (σ = rms_y); time source as stated. I-DT uses the app default "
         f"idt_k = {cfg['analysis']['idt_k']} unless stated.",
         "- `*` = layout does not fit the 1512×945 page with whole paragraphs; excluded from thresholds.", "",
         "## Minimum line pitch for line-level ρ ≥ 0.7", "",
         "| σ source | σ_y (px) | time source | min pitch (within-para ρ) | min pitch (pooled ρ) | line accuracy at 66 px | ρ_para at 66 px |",
         "|---|---|---|---|---|---|---|"]
    for r in gl:
        L.append(f"| {r['sigma_label']} | {r['sigma_y']} | {r['source']} | {r['min_pitch_within_rho_0.7']} | "
                 f"{r['min_pitch_pooled_rho_0.7']} | {r['acc_line@66']:.2f} | {r['rho_para@66']:.2f} |")
    L += ["", "Within-paragraph line ρ by pitch (samples):", "",
          "| σ source | " + " | ".join(k.split("@")[1] for k in gl[0] if k.startswith("within_rho@")) + " |",
          "|---|" + "---|" * sum(k.startswith("within_rho@") for k in gl[0])]
    for r in gl:
        if r["source"] == "samples":
            L.append(f"| {r['sigma_label']} | " + " | ".join(f"{v:.2f}" for k, v in r.items() if k.startswith("within_rho@")) + " |")
    L += ["", f"## Minimum paragraph height for paragraph-level ρ ≥ 0.7 (pitch {base:g} px, 6 paragraphs, samples)", "",
          "| σ source | σ_y (px) | min lines per paragraph | " + " | ".join(k.split("@")[1] for k in ph[0] if k.startswith("rho_para@")) + " |",
          "|---|---|---|" + "---|" * sum(k.startswith("rho_para@") for k in ph[0])]
    for r in ph:
        L.append(f"| {r['sigma_label']} | {r['sigma_y']} | {r['min_para_lines_rho_0.7']} | "
                 + " | ".join(f"{v:.2f}" for k, v in r.items() if k.startswith("rho_para@")) + " |")
    L += ["", f"## I-DT threshold sweep (pitch {base:g} px, fixations source)", "",
          "Detected fixation time as a share of true fixation time, and within-paragraph line ρ:", "",
          "| σ_y | " + " | ".join(f"k={k:g} time" for k in cfg["grid"]["idt_k"]) + " | "
          + " | ".join(f"k={k:g} ρ" for k in cfg["grid"]["idt_k"]) + " |",
          "|---|" + "---|" * (2 * len(cfg["grid"]["idt_k"]))]
    for row in idt_rows:
        L.append(f"| {row['sigma_y']:g} | " + " | ".join(f"{row[f'time_{k:g}']:.2f}" for k in cfg["grid"]["idt_k"])
                 + " | " + " | ".join(f"{row[f'rho_{k:g}']:.2f}" for k in cfg["grid"]["idt_k"]) + " |")
    L += ["", f"## Ablations: within-paragraph line ρ vs σ_y at pitch {base:g} px", "",
          "| σ_y | fixations (soft, filter) | hard mapping | no filter | samples |", "|---|---|---|---|---|"]
    for sy in cfg["grid"]["sigma_y"]:
        get = lambda **kw: sel(agg, experiment="grid", pitch=base, sigma_y=sy, **dict(PRIMARY, **kw))[0]["rho_line_within"]
        L.append(f"| {sy:g} | {get():.2f} | {get(mapping='hard'):.2f} | {get(filter=0):.2f} | {get(source='samples'):.2f} |")
    L += ["", "Limitations: simplified reading model (character grid, no word lengths or difficulty); noise "
          "assumed stationary within a reading; σ from one person's live calibrations; the analysis is given the "
          "true σ (a perfectly estimated calibration error).", ""]
    return "\n".join(L)


def idt_table(agg, cfg) -> list[dict]:
    out = []
    for sy in cfg["grid"]["sigma_y"]:
        row = {"sigma_y": sy}
        for k in cfg["grid"]["idt_k"]:
            r = sel(agg, experiment="idt", sigma_y=sy, idt_k=k, filter=1, mapping="soft", source="fixations")[0]
            row[f"time_{k:g}"], row[f"rho_{k:g}"] = r["fix_time_frac"], r["rho_line_within"]
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, help="override n_seeds (e.g. 3 for a quick check)")
    args = ap.parse_args()
    cfg = load_cfg()
    if args.seeds:
        cfg["n_seeds"] = args.seeds
    FIGS.mkdir(parents=True, exist_ok=True)
    ts = tasks(cfg, cfg["n_seeds"])
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 1)) as ex:
        for r in ex.map(run_one, ts, chunksize=8):
            rows += r
    runtime = time.time() - t0
    agg = aggregate(rows)
    write_csv(TABLES / "sim_results.csv", agg)
    gl, ph = guideline_tables(agg, cfg)
    write_csv(TABLES / "sim_guideline.csv", gl)
    write_csv(TABLES / "sim_para_height.csv", ph)
    idt_rows = idt_table(agg, cfg)
    write_csv(TABLES / "sim_idt_sweep.csv", idt_rows)
    figs = figures(agg, cfg)
    (ROOT / "results" / "sim_summary.md").write_text(summary_md(agg, gl, ph, idt_rows, cfg, runtime, len(ts)),
                                                     encoding="utf-8")
    print(f"{len(ts)} readings in {runtime:.0f} s -> {TABLES}, {len(figs)} figures, results/sim_summary.md")


if __name__ == "__main__":
    main()
