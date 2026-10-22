"""M5: leave-one-person-out evaluation on MPIIFaceGaze with simulated k-shot calibration (spec 7.1).

    python -m gazeread.eval.lopo        # `make eval-lopo`, after `make extract-mpii`

For each held-out person p and each variant (feature set):
  (i)  global + per-user correction: a ridge map features -> 2D target (mm) fitted on the other
       people only (degree and alpha chosen by person-wise inner CV over those people), then
       corrected with p's k calibration frames: k=0 none, k=1 offset, k>=4 ridge affine on the
       global output (alpha by LOO over the k points).
  (ii) per-user only: the live app's calibration (calibration.select_model) on p's k frames.
Calibration frames come from select_kshot_points over p's *valid* frames; every other valid
frame of p is evaluated. Errors are Euclidean on the screen plane in mm, also as % of p's screen
height and as equivalent CSS px on the user's own display (configs/eval_lopo.yaml: my_screen).
"""
import csv
import time
from pathlib import Path

import numpy as np
import yaml

from gazeread.eval.kshot import select_kshot_points
from gazeread.gaze.calibration import RidgeMap, Scaler, select_model

ROOT = Path(__file__).resolve().parents[3]
TABLES, FIGS = ROOT / "results" / "tables", ROOT / "results" / "figures"


def load_cfg() -> dict:
    return yaml.safe_load((ROOT / "configs/eval_lopo.yaml").read_text())


def load_people(cfg: dict) -> dict:
    """{person: {"landmark": (n,12), "l2cs": (n,4), "valid": (n,), "target_mm": (n,2),
    "target_frac": (n,2), "screen_mm": (w,h)}} from the cached extraction."""
    people = {}
    for f in sorted((ROOT / cfg["processed_dir"] / "mpii").glob("p*.npz")):
        d = np.load(f, allow_pickle=False)
        people[f.stem] = {k: d[k] for k in d.files}
    return people


def person_folds(people: list[str]):
    """Yield (held_out, training_people); every person is held out exactly once."""
    for p in people:
        yield p, [q for q in people if q != p]


def _alphas(spec):
    return np.logspace(spec[0], spec[1], int(spec[2]))


def fit_global(X_by_person: list[np.ndarray], Y_by_person: list[np.ndarray], gcfg: dict) -> tuple[RidgeMap, dict]:
    """Ridge map on the training people only; degree/alpha by person-wise inner CV."""
    n = len(X_by_person)
    folds = np.array_split(np.arange(n), min(gcfg["inner_folds"], n))
    best = {}
    for deg in gcfg["degrees"]:
        for a in _alphas(gcfg["alphas_log10"]):
            errs = []
            for f in folds:
                tr = [i for i in range(n) if i not in f]
                Xtr, Ytr = np.vstack([X_by_person[i] for i in tr]), np.vstack([Y_by_person[i] for i in tr])
                m = RidgeMap(deg, a, Scaler.fit(Xtr)).fit(Xtr, Ytr)
                for i in f:
                    errs.append(np.linalg.norm(m.predict(X_by_person[i]) - Y_by_person[i], axis=1).mean())
            best[(deg, a)] = float(np.mean(errs))
    b1 = min((v, k) for k, v in best.items() if k[0] == 1)
    choice = b1
    if 2 in gcfg["degrees"]:
        b2 = min((v, k) for k, v in best.items() if k[0] == 2)
        if b2[0] < (1 - gcfg["poly2_margin"]) * b1[0]:
            choice = b2
    deg, a = choice[1]
    X, Y = np.vstack(X_by_person), np.vstack(Y_by_person)
    return RidgeMap(deg, a, Scaler.fit(X)).fit(X, Y), {"degree": deg, "alpha": a, "inner_cv_mm": choice[0]}


def corrected(G_cal: np.ndarray, Y_cal: np.ndarray, G_eval: np.ndarray, k: int, alphas) -> np.ndarray:
    """Per-user correction of global predictions from k calibration frames.

    k=1..3: constant offset. k>=4: the residual Y - G is modelled as offset + B G with ridge on B,
    i.e. regularised towards the identity, so heavy regularisation degrades to the offset-only
    correction instead of collapsing onto the mean calibration target. alpha by LOO over the k points.
    """
    if k == 0:
        return G_eval
    if k < 4:
        return G_eval + (Y_cal - G_cal).mean(0)
    m, _ = select_model(G_cal, Y_cal - G_cal, degrees=(1,), alphas=alphas)
    return G_eval + m.predict(G_eval)


def evaluate(people: dict, cfg: dict, variants: dict) -> tuple[list[dict], list[dict], dict]:
    """Returns per-(variant, strategy, k, person) rows, per-frame error samples for the screen
    heatmap, and the global-model choices per fold."""
    rows, frame_err, choices = [], [], {}
    names = sorted(people)
    for vname, (backend, cols) in variants.items():
        for p, train in person_folds(names):
            Xs = [people[q][backend][people[q]["valid"]][:, cols] for q in train]
            Ys = [people[q]["target_mm"][people[q]["valid"]] for q in train]
            G, info = fit_global(Xs, Ys, cfg["global"])
            choices[(vname, p)] = info
            P = people[p]
            vidx = np.flatnonzero(P["valid"])
            X, Y = P[backend][vidx][:, cols], P["target_mm"][vidx]
            G_all = G.predict(X)
            h_mm = float(P["screen_mm"][1])
            for k in cfg["k_values"]:
                for s in range(cfg["n_seeds"] if k > 0 else 1):
                    cal = select_kshot_points(Y, k, cfg["seed"] * 1000 + s) if k > 0 else np.array([], int)
                    ev = np.setdiff1d(np.arange(len(vidx)), cal)
                    assert not np.intersect1d(cal, ev).size            # no calibration frame is evaluated
                    preds = {"global+correction": corrected(G_all[cal], Y[cal], G_all[ev], k, _alphas(cfg["correction"]["alphas_log10"]))}
                    if k >= cfg["per_user_min_k"]:
                        m, _ = select_model(X[cal], Y[cal])
                        preds["per-user"] = m.predict(X[ev])
                    for strat, pr in preds.items():
                        e = np.linalg.norm(pr - Y[ev], axis=1)
                        rows.append({"variant": vname, "strategy": strat, "k": k, "seed": s, "person": p,
                                     "err_mm": float(e.mean()), "err_pct_h": float(100 * e.mean() / h_mm),
                                     "n_eval": int(len(ev))})
                        if strat == "global+correction" and k == 9 and s == 0:
                            frame_err.append((vname, P["target_frac"][vidx][ev], e))
    return rows, frame_err, choices


def summarize(rows: list[dict], people: dict, cfg: dict) -> tuple[list[dict], list[dict], list[dict]]:
    px_per_mm = cfg["my_screen"]["width_px"] / cfg["my_screen"]["width_mm"]
    # per person: mean over seeds
    per_person = {}
    for r in rows:
        key = (r["variant"], r["strategy"], r["k"], r["person"])
        per_person.setdefault(key, []).append(r)
    pp_rows = [{"variant": v, "strategy": st, "k": k, "person": p,
                "err_mm": float(np.mean([r["err_mm"] for r in rs])),
                "err_pct_h": float(np.mean([r["err_pct_h"] for r in rs])),
                "excluded_frac": float(1 - people[p]["valid"].mean())}
               for (v, st, k, p), rs in sorted(per_person.items())]
    table = []
    groups = {}
    for r in pp_rows:
        groups.setdefault((r["variant"], r["strategy"], r["k"]), []).append(r)
    flag = cfg["excluded_flag_frac"]
    for (v, st, k), rs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        e = np.array([r["err_mm"] for r in rs])
        ok = np.array([r["excluded_frac"] <= flag for r in rs])
        table.append({"variant": v, "strategy": st, "k": k, "n_people": len(rs),
                      "err_mm_mean": round(float(e.mean()), 1), "err_mm_sd": round(float(e.std(ddof=1)), 1),
                      "err_pct_h_mean": round(float(np.mean([r["err_pct_h"] for r in rs])), 1),
                      "err_mm_worst": round(float(e.max()), 1),
                      "my_screen_px_mean": round(float(e.mean() * px_per_mm), 0),
                      "err_mm_mean_unflagged": round(float(e[ok].mean()), 1) if ok.any() else float("nan")})
    excl = [{"person": p, "excluded_frac": round(float(1 - people[p]["valid"].mean()), 3),
             "n_frames": int(len(people[p]["valid"])), "flagged": int(1 - people[p]["valid"].mean() > flag)}
            for p in sorted(people)]
    return table, pp_rows, excl


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def figures(table, pp_rows, frame_err, cfg) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, muted, grid_c = "#1f1f1e", "#6b6a64", "#e6e4dd"
    cat = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": grid_c, "grid.linewidth": 0.8, "lines.linewidth": 2})
    paths = []
    variants = list(cfg["variants"])

    # 1. error vs k, both strategies
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, strat in zip(axes, ("global+correction", "per-user")):
        for c, v in zip(cat, variants):
            rs = sorted([r for r in table if r["variant"] == v and r["strategy"] == strat], key=lambda r: r["k"])
            if not rs:
                continue
            ks = [r["k"] for r in rs]
            m = np.array([r["err_mm_mean"] for r in rs]); sd = np.array([r["err_mm_sd"] for r in rs])
            ax.plot(ks, m, color=c, marker="o", ms=5, label=v)
            ax.fill_between(ks, m - sd, m + sd, color=c, alpha=0.12, lw=0)
        ax.set_title(f"{strat}", loc="left", color=ink, fontsize=11)
        ax.set_xlabel("calibration points k")
        ax.set_xticks(cfg["k_values"])
        ax.legend(frameon=False, fontsize=9, labelcolor=ink)
    axes[0].set_ylabel("mean error on screen (mm), ±1 sd over people")
    fig.suptitle("MPIIFaceGaze, leave-one-person-out: error vs number of calibration points", x=0.01, ha="left",
                 color=ink, fontsize=12)
    fig.tight_layout()
    paths.append(FIGS / "m5_error_vs_k.png"); fig.savefig(paths[-1], dpi=150); plt.close(fig)

    # 2. per-participant bars at k = 9 (global+correction), headline variants
    fig, ax = plt.subplots(figsize=(12, 4.2))
    heads = cfg["headline"]
    people = sorted({r["person"] for r in pp_rows})
    w = 0.8 / len(heads)
    for i, (c, v) in enumerate(zip(cat, heads)):
        vals = [next(r["err_mm"] for r in pp_rows if r["variant"] == v and r["strategy"] == "global+correction"
                     and r["k"] == 9 and r["person"] == p) for p in people]
        ax.bar(np.arange(len(people)) + (i - (len(heads) - 1) / 2) * w, vals, width=w * 0.92, color=c, label=v)
    flags = {r["person"] for r in pp_rows if r["excluded_frac"] > cfg["excluded_flag_frac"]}
    ax.set_xticks(np.arange(len(people)), [p + (" *" if p in flags else "") for p in people])
    ax.set_ylabel("mean error (mm), k = 9")
    ax.set_title("Per participant (global + correction, k = 9); * = more than 15% frames excluded",
                 loc="left", color=ink, fontsize=11)
    ax.legend(frameon=False, fontsize=9, labelcolor=ink)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    paths.append(FIGS / "m5_per_participant.png"); fig.savefig(paths[-1], dpi=150); plt.close(fig)

    # 3. error over screen position (k = 9, seed 0), headline variants
    fig, axes = plt.subplots(1, len(heads), figsize=(6 * len(heads), 4))
    nx, ny = 6, 4
    for ax, v in zip(np.atleast_1d(axes), heads):
        S = np.zeros((ny, nx)); N = np.zeros((ny, nx))
        for vn, frac, e in frame_err:
            if vn != v:
                continue
            ix = np.clip((frac[:, 0] * nx).astype(int), 0, nx - 1)
            iy = np.clip((frac[:, 1] * ny).astype(int), 0, ny - 1)
            np.add.at(S, (iy, ix), e); np.add.at(N, (iy, ix), 1)
        M = np.where(N > 0, S / np.maximum(N, 1), np.nan)
        im = ax.imshow(M, cmap="Blues", origin="upper", extent=(0, 1, 1, 0))
        for iy in range(ny):
            for ix in range(nx):
                if N[iy, ix]:
                    ax.text((ix + 0.5) / nx, (iy + 0.5) / ny, f"{M[iy, ix]:.0f}", ha="center", va="center",
                            fontsize=8, color=ink if M[iy, ix] < np.nanmax(M) * 0.6 else "white")
        ax.set_title(f"{v}: mean error (mm) by target position", loc="left", color=ink, fontsize=11)
        ax.set_xlabel("screen x (fraction of width)"); ax.set_ylabel("screen y (fraction of height)")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.04)
    fig.tight_layout()
    paths.append(FIGS / "m5_error_by_screen_position.png"); fig.savefig(paths[-1], dpi=150); plt.close(fig)
    return paths


def summary_md(table, excl, choices, cfg, runtime) -> str:
    flag = cfg["excluded_flag_frac"]
    L = ["# M5 MPIIFaceGaze leave-one-person-out summary (generated by `make eval-lopo`; do not edit by hand)", "",
         f"- {len(excl)} participants, {sum(r['n_frames'] for r in excl)} frames; {cfg['n_seeds']} k-shot seeds; "
         f"runtime {runtime:.0f} s.",
         "- Errors: Euclidean on the screen plane, mean over each person's evaluated frames, then mean ± sd over people. "
         "`my px` = the mean converted to CSS px on the user's display "
         f"({cfg['my_screen']['width_mm']} mm = {cfg['my_screen']['width_px']} px), comparable with live σ_px "
         "(which is an RMS, so typically larger than a mean for the same errors).",
         "- Calibration frames are never evaluated (asserted in code and tested).", "",
         "## Error vs k (headline variants)", "",
         "| variant | strategy | k | mean mm | sd mm | % screen h | my px | worst person mm | mean mm excl. flagged |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in table:
        if r["variant"] in cfg["headline"]:
            L.append(f"| {r['variant']} | {r['strategy']} | {r['k']} | {r['err_mm_mean']} | {r['err_mm_sd']} | "
                     f"{r['err_pct_h_mean']} | {r['my_screen_px_mean']:.0f} | {r['err_mm_worst']} | {r['err_mm_mean_unflagged']} |")
    L += ["", "## All variants at k = 9 (global + correction)", "", "| variant | mean mm | sd mm | % screen h | my px |",
          "|---|---|---|---|---|"]
    for r in table:
        if r["k"] == 9 and r["strategy"] == "global+correction":
            L.append(f"| {r['variant']} | {r['err_mm_mean']} | {r['err_mm_sd']} | {r['err_pct_h_mean']} | {r['my_screen_px_mean']:.0f} |")
    L += ["", "## Excluded frames (no face, eye closed or extreme head pose; same gate for A and B)", "",
          "| person | frames | excluded | flagged (> 15%) |", "|---|---|---|---|"]
    for r in excl:
        L.append(f"| {r['person']} | {r['n_frames']} | {100 * r['excluded_frac']:.1f}% | {'**yes**' if r['flagged'] else ''} |")
    flagged = [r["person"] for r in excl if r["flagged"]]
    L += ["", ("Flagged: " + ", ".join(flagged) + f". Their errors are computed only on the frames that passed the "
               "validity gate, which are typically the easier ones (frontal, eyes open), so their numbers are "
               "optimistic; the table also gives each mean without them.") if flagged else
          f"No participant has more than {100 * flag:.0f}% excluded frames.", "",
          "## Global model chosen per fold (inner person-wise CV over the 14 training people)", "",
          "| variant | degrees chosen | alphas chosen |", "|---|---|---|"]
    for v in cfg["variants"]:
        cs = [c for (vn, _), c in choices.items() if vn == v]
        L.append(f"| {v} | {sorted({int(c['degree']) for c in cs})} | {sorted({float(round(c['alpha'], 4)) for c in cs})} |")
    return "\n".join(L) + "\n"


def main() -> None:
    cfg = load_cfg()
    people = load_people(cfg)
    t0 = time.time()
    rows, frame_err, choices = evaluate(people, cfg, cfg["variants"])
    table, pp_rows, excl = summarize(rows, people, cfg)
    TABLES.mkdir(parents=True, exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)
    write_csv(TABLES / "m5_error_vs_k.csv", table)
    write_csv(TABLES / "m5_per_participant.csv", pp_rows)
    write_csv(TABLES / "m5_excluded_frames.csv", excl)
    figs = figures(table, pp_rows, frame_err, cfg)
    runtime = time.time() - t0
    (ROOT / "results" / "m5_summary.md").write_text(summary_md(table, excl, choices, cfg, runtime), encoding="utf-8")
    print(f"LOPO done in {runtime:.0f} s -> {TABLES}, {len(figs)} figures, results/m5_summary.md")


if __name__ == "__main__":
    main()
