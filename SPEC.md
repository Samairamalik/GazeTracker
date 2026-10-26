# GazeRead: webcam gaze-based attention heatmaps for reading

This file is the single source of truth for this project. Read it fully before writing code. If something here conflicts with what you find in the repo or the data, stop and tell me; do not silently pick one.

---

## 0. Working rules (merge with my standard template)

> My standard template sections are below in short form. If I paste my own wording, it overrides these.

### Think before coding
- State your plan and assumptions before implementing anything non-trivial. If a requirement is ambiguous, ask once, concisely, rather than guessing.
- Before using any external API, file format or landmark index, verify it against the real docs, README or an actual debug visualization. Several items in this file are marked **VERIFY**. Do not trust them blindly.
- Prefer Plan mode for each milestone. Show me the plan, then build.

### Simplicity first
- Smallest thing that satisfies the acceptance criteria. No frameworks, abstractions or config systems "for later".
- Plain Python modules, dataclasses, numpy, scikit-learn, PyTorch. No unnecessary dependencies.
- Frontend is vanilla HTML/JS. No build step, no npm, no React.

### Surgical changes
- Touch only what the current milestone needs. Do not reformat or refactor unrelated files.
- Every change should be explainable in one sentence tied to a milestone.

### Goal-driven execution
- Every milestone below has acceptance criteria. Work toward them, run the checks, and report real numbers.
- Never say "done" or "works" without having run it. If you could not run something (e.g. needs a webcam, a human, or a dataset I have not downloaded yet), say so explicitly.
- Report results honestly. If an approach performs badly, report that. Do not tune on the test person or cherry-pick.

---

## 1. Project in one paragraph

Build a system that uses an ordinary laptop webcam to estimate where on screen a person is looking while reading text, and turns that into a heatmap of which parts of the text received attention. It is the final project for my Principles of Computer Vision course (team of 2). **We build the working application first** (live calibration, gaze cursor, reading page, fixation detection, attention heatmaps). A **secondary, later evaluation on a public dataset** (MPIIFaceGaze) then measures how accurate the system really is across many people. The dataset is never needed to build or run the app. Possible uses: ed-tech, UX research.

### Honest framing (do not oversell in code comments, README or report)
- Webcam gaze is coarse. Typical consumer-webcam error is a few degrees of visual angle. At about 60 cm viewing distance, 4° is roughly 4 cm on screen, which can be 2 or more lines of normal-size text. L2CS-Net reports about 3.9° mean angular error on MPIIGaze in leave-one-person-out evaluation, and that is the benchmark setting, not our live setting.
- Therefore the **default output is paragraph-level, with line-level only when validated**. The app must choose its aggregation granularity from the measured calibration error (section 6.5), not hard-code it.
- Word-level heatmaps are a stretch goal and probably infeasible with a webcam. Do not build toward them unless asked.

---

## 2. Hard constraints

1. **No self-collected dataset.** Train and evaluate on public datasets only. Live webcam use is allowed for the demo and for per-user calibration at runtime, but we do not collect, store or publish a dataset of faces.
2. **Privacy by default.** Do not save webcam video or face images to disk unless a `--save-eye-crops` style flag is explicitly passed. Session logs store landmark-derived features, gaze coordinates and events only. Add a one-line consent notice at app start.
3. **Platform: macOS** (Apple Silicon assumed, verify with `uname -m`). Use `torch.device("mps")` when available, fall back to CPU. No CUDA-only code. Camera access on macOS needs permission granted to the terminal app running Python; detect failure to open the camera and print a helpful message.
4. **Python 3.11**, virtual environment via `uv` or `venv`. Pin versions in `requirements.txt`. Check the installed `mediapipe` version has a macOS wheel for the Python version in use (**VERIFY**).
5. **Reproducibility:** fixed seeds, all experiment parameters in `configs/*.yaml`, every results table or figure regenerable with one command (`make results` or a script).
6. **Licensing/citation:** MPIIFaceGaze and Gaze360 are research-use datasets. Do not commit data or weights. Keep a `CITATIONS.md` with MPIIFaceGaze, L2CS-Net, Gaze360 (if used) and MediaPipe.
7. Team of 2: module boundaries (section 5) must let two people work in parallel. The interface between them is the `GazePoint` stream plus the session log format (section 4.3).

---

## 3. Datasets and models

### 3.1 Evaluation dataset (secondary, only needed at M5): MPIIFaceGaze
- Laptop-webcam, full-face images with on-screen gaze targets, about 37,667 images from 15 participants (**VERIFY** participant count from the README). Annotation includes the gaze location on screen in pixels, 3D gaze target in camera coordinates, facial landmarks, and per-participant screen size (pixels and mm) in a calibration folder.
- Download is manual from the Max Planck / perceptualui.org MPIIFaceGaze page (original release about 940 MB; a normalized 448×448 version also exists). **Do not ask me for it, and do not mention it, until we reach M5. Then ask me to download it; never try to scrape it.** Put it under `data/raw/MPIIFaceGaze/` (gitignored).
- **VERIFY the exact annotation file layout** by reading the dataset's `readme.txt` and printing the first lines of one annotation file before writing the loader. Write the layout you confirmed as a comment at the top of `data/mpiifacegaze.py`.
- Evaluation protocol is **leave-one-person-out (LOPO)**, the same protocol the dataset authors use. A person in the test fold must never appear in training or in hyperparameter selection for that fold.

### 3.2 Gaze models
- **Method A: landmark features + regression (classical baseline).** MediaPipe FaceLandmarker (Tasks API: `mediapipe.tasks.python.vision.FaceLandmarker`, with `output_facial_transformation_matrixes=True`) gives 478 landmarks including iris, plus a head transformation matrix. Hand-crafted features, then ridge or low-order polynomial regression to screen coordinates. No deep training required.
- **Method B: pretrained L2CS-Net.** Predicts gaze direction (yaw, pitch) from a face crop with a ResNet50 backbone. Upstream repo: `Ahmednull/L2CS-Net` (MIT). Install via `pip install git+https://github.com/Ahmednull/L2CS-Net.git@main` or clone and `pip install -e .`. Pretrained weights (e.g. `L2CSNet_gaze360.pkl`) are linked from the upstream README, usually on Google Drive. **Ask me to download the weights manually** and place them in `models/` (gitignored). Prefer feeding it the face crop from the MediaPipe bounding box rather than running a second face detector, unless that hurts accuracy; if you try both, report it.
- **Method C (optional, only if A and B are done):** fine-tune a small CNN (ResNet18) on MPIIFaceGaze under LOPO or 5-fold-by-person. Likely compute-limited on a Mac; skip if it is not clearly feasible.

### 3.3 The key design idea: global model + tiny per-user calibration
Public data cannot remove the need for a short calibration at deployment, because every screen and camera geometry is different. So the whole pipeline is:

```
frame -> face features / gaze angles  --(global model, no user data needed)-->
   raw gaze estimate  --(per-user calibration map, 9-16 dots)-->  screen (x, y)
```

The per-user calibration map is a low-capacity model (affine, or 2nd-order polynomial with ridge) from the raw estimate to screen coordinates. We evaluate this honestly on MPIIFaceGaze by **simulating k-shot calibration for each held-out person** (section 7.1). That produces the project's headline result: error as a function of the number of calibration points, for each method.

---

## 4. Features and the data contract

### 4.1 Landmark features (Method A)
Per frame, compute and return a fixed-length vector. Document the order in code.

- For each eye: iris center expressed in eye-local coordinates. `u` = projection of (iris center − eye center) on the inner-to-outer corner axis, divided by eye width. `v` = the perpendicular component, divided by eye width.
- Eye openness per eye (lid distance / eye width) and a `valid` flag that is false when the eye is closed (blink), the face is missing, or the head pose is extreme.
- Head pose from the facial transformation matrix: yaw, pitch, roll and the 3 translation components.
- Optional extras if they help in cross-validation: face-box center and scale in the image.

MediaPipe indices commonly used with the 478-point model (**VERIFY by drawing the indices on a frame and viewing it; do not trust this list**):
- Right-eye iris: center 468, ring 469–472. Left-eye iris: center 473, ring 474–477.
- Eye corners: right eye 33 (outer) and 133 (inner); left eye 263 (outer) and 362 (inner).
- Lids: right eye 159 (upper) and 145 (lower); left eye 386 (upper) and 374 (lower).

### 4.2 Core dataclasses (put in `src/gazeread/types.py`; everything else imports these)
```python
@dataclass
class FrameFeatures:
    t: float                 # seconds, monotonic, from capture time
    valid: bool              # False on blink / no face / extreme pose
    vec: np.ndarray          # feature vector (Method A) or [yaw, pitch] (Method B)

@dataclass
class GazePoint:
    t: float
    x: float                 # viewport CSS px, origin top-left of the reading viewport
    y: float
    valid: bool

@dataclass
class Fixation:
    t0: float; t1: float
    x: float; y: float       # centroid
    @property
    def dur(self): return self.t1 - self.t0

@dataclass
class LineBox:
    page: int; line_id: int; para_id: int
    x0: float; y0: float; x1: float; y1: float   # viewport CSS px
    text: str
```
All screen coordinates are **viewport CSS pixels in a fullscreen browser window**. Calibration dots and text line boxes live in the same coordinate frame, which avoids screen/window offset bugs. Account for `devicePixelRatio` on Retina displays and record it in the session log.

### 4.3 Session log format (JSONL, one event per line, in `sessions/<id>/events.jsonl`, gitignored)
Event types: `meta` (screen size, devicePixelRatio, fps, config hash, `source`: camera|synthetic, primary `backend`, `compare`, `variants`, reading `text`, `skim` (deliberately skimmed passage); synthetic sessions are test runs and are never reported as accuracy results), `calib_target` (x, y, t_on, t_off, `phase`: fit|validate), `calib_result` (ok, `variant` A|B|B+pose, `backend`, `primary` (drives the cursor; compare mode writes one per variant from the same dots), n_points, features, degree, alpha, loo_rms_px, sigma_px, rms_x, rms_y, per-dot median error; or ok=false with a reason), `features` (t, valid, vec, `backend`: landmark|l2cs; missing = landmark), `gaze` (t, x, y, valid), `page` (page index, t_on stamped by Python on receipt, line boxes, line_pitch, font_px), `reading_end` (t; closes the last page of a reading), `answer` (no longer produced: the quiz was removed after the M6 human check; old logs keep it) (t, passage, question_id, para_id = paragraph the question is about, response = option index, correct; scored by the server from `texts/<passage>.questions.json`, the page never receives the answer key). Analysis scripts must work from logs alone, so any experiment can be re-run offline without the webcam.

---

## 5. Architecture and repo layout

```
gazeread/
  SPEC.md  README.md  CITATIONS.md  requirements.txt  Makefile
  configs/            default.yaml, eval_lopo.yaml, sim.yaml
  data/               (gitignored) raw/, processed/
  models/             (gitignored) L2CS weights
  sessions/           (gitignored) live-session logs
  results/            figures/, tables/ (committed: small csv + png)
  notebooks/          exploration only; no logic that the pipeline depends on
  scripts/            check_env.py, debug_landmarks.py
  src/gazeread/
    types.py
    data/mpiifacegaze.py        loader, per-person splits, px<->mm conversion
    features/landmarks.py       MediaPipe wrapper -> FrameFeatures
    features/head_pose.py
    gaze/backend_landmark.py    Method A feature extractor
    gaze/backend_l2cs.py        Method B wrapper
    gaze/calibration.py         fit/apply per-user map; k-shot point selection
    gaze/filters.py             One Euro filter (and moving average for comparison)
    analysis/fixations.py       I-DT detector
    analysis/line_mapping.py    hard + soft assignment of fixations to lines/paragraphs
    analysis/heatmap.py         dwell aggregation + rendering
    sim/scanpath.py             synthetic reading scanpaths with noise
    eval/lopo.py  eval/kshot.py  eval/metrics.py  eval/sim_experiments.py
    app/server.py               FastAPI + WebSocket
    app/session.py              logging
    app/static/                 index.html, calibrate.js, reader.js, heatmap.js
  tests/
```

Runtime split for the live app:
- **Python process:** owns the webcam (OpenCV `VideoCapture`, 30 fps), runs the feature extractor, applies the calibration map and filter, streams `GazePoint`s over WebSocket, writes the session log.
- **Browser page (fullscreen):** shows calibration dots, the reading text, and the final heatmap. It sends `calib_target` events and per-page `LineBox` geometry (via `getBoundingClientRect` on line spans) to Python.
- Reading text is **paged, not scrolled** (about 12–16 lines per page, advance with spacebar). This removes scroll-to-gaze mapping.
- Reading style defaults that help gaze accuracy: large font (28–32 px), line pitch of at least 2× font size, wide margins, single column, plain background. Make these config values and evaluate line-pitch effects in the simulation (section 7.3).

---

## 6. Algorithms: details to implement

### 6.1 Calibration (`gaze/calibration.py`)
- Targets: 9-point (3×3) default, 16-point (4×4) option, inset from edges by about 8% of viewport. Each dot shown about 1.5 s; discard the first 400 ms (saccade latency) and last 100 ms, use the median of remaining valid features for that dot, not every frame.
- Model: ridge regression from raw estimate (Method A feature vector or Method B yaw/pitch) to `(x, y)`. Compare affine vs degree-2 polynomial; choose regularization by leave-one-dot-out CV. With only 9–16 points, **default to the lowest-capacity model that validates well**.
- **Validation step:** after fitting, show 5 fresh dots not used in fitting, compute error in px, and store `sigma_px` (RMS error). This `sigma_px` drives smoothing strength, fixation dispersion threshold and aggregation granularity (6.5).
- Provide `select_kshot_points(positions, k, seed)` for evaluation: choose k calibration frames spread across the screen (grid-cell or k-means on target positions), never random-clumped.

### 6.2 Filtering (`gaze/filters.py`)
- One Euro filter on (x, y), with parameters in config. Provide a moving average for ablation. Handle invalid frames by holding for at most 150 ms then marking gaps.

### 6.3 Fixation detection (`analysis/fixations.py`)
- I-DT: window over valid samples; dispersion (max−min x plus max−min y) below a threshold and duration at least 100–150 ms. Set the dispersion threshold from `sigma_px` (starting point about 1.0–1.5 × sigma_px); expose it as config. Merge fixations separated by under 75 ms and near each other.
- At about 30 fps this gives 3–5 samples per minimal fixation; be explicit about this limitation in the report.

### 6.4 Line and paragraph mapping (`analysis/line_mapping.py`)
- **Hard assignment:** nearest line box by vertical distance (with a tolerance).
- **Soft assignment (preferred):** distribute each fixation's duration across lines with Gaussian weights in y, using `sigma_y` equal to the calibration's vertical RMS error `rms_y` (the Euclidean `sigma_px` overstates the vertical spread). This avoids brittle line flipping when error exceeds line pitch.
- **Two time sources:** credit time from I-DT fixations and, alternatively, from every gaze sample (`samples` mode, each sample's time to the next sample). Both are always reported so they can be compared in M4; `analysis.mapping_source` picks the primary view.
- Output dwell time per line and per paragraph, normalized per page.

### 6.5 Heatmap (`analysis/heatmap.py`, `static/heatmap.js`)
- Overlay a color gradient on each line (and a paragraph-level band). Also provide a scanpath view (fixation dots sized by duration, joined by lines). Export PNG and a JSON of dwell values.
- Auto granularity (see 6.1 and 1): if `rms_y` is more than half the line pitch, show paragraph-level as primary and label line-level as "low confidence".

---

## 7. Evaluation protocol (7.3 runs before the app is finished; 7.1 is the secondary, last step)

### 7.1 Gaze accuracy on MPIIFaceGaze (offline, no webcam needed)
For each method (A, B, optionally C):
1. LOPO: for each held-out participant, fit any global component on the other participants only.
2. Simulated calibration: for the held-out person select k ∈ {1, 4, 9, 16, 25} calibration frames via `select_kshot_points`, fit the per-user map, evaluate on all remaining frames of that person. Repeat over several seeds.
3. Report: mean and std over participants, in **mm on the screen plane** (via each participant's screen size) and as a fraction of screen height. Also report per-participant results (some are far worse; show them).
4. Figures: error vs k curves for each method; per-participant bar chart; error heatmap over screen position (is the error worse near edges and corners?); ablation of features (iris only vs iris+head pose) and of regressors (affine, poly2, ridge strengths).
5. Frames with invalid features are excluded from error stats but the **fraction excluded** is reported per method.

### 7.2 Live system validation
Hold-out validation dots (6.1). Log `sigma_px` per session. Report for at least 2 people and a few sessions. This is a demonstration, not a dataset.

### 7.3 Reading-behavior simulation (replaces "film ourselves reading for 20 minutes")
Use `sim/scanpath.py` to generate synthetic reading over a real `LineBox` layout: fixations advance left to right along each line (saccade lengths around 7–9 characters, fixation durations around 200–250 ms with variation), return sweeps between lines, occasional regressions, and some skimmed paragraphs. Ground-truth dwell per line and paragraph is known. Then add measurement noise matching the measured error (Gaussian, plus a slow drift term, with sigma from the live validation in 7.2, and from 7.1 once it exists) and sample at 30 fps.
Experiments:
- Line-assignment accuracy and Spearman correlation between true and estimated dwell per line, versus noise sigma and versus line pitch (e.g. 50, 64, 80, 100 px).
- Paragraph-level versions of the same.
- Hard vs soft assignment; with vs without smoothing.
- **Design guideline output:** the minimum line pitch / font size at which line-level heatmaps reach a target correlation (say 0.7) at the measured sigma.

### 7.4 Optional human-in-the-loop check
Two of us read 3–4 short passages (public-domain or Wikipedia text, 150–300 words each) with 4–5 comprehension questions each, deliberately skimming one passage. Check qualitatively and with a simple statistic whether low-attention paragraphs correlate with missed questions. With n=2 this is illustrative only; say so.

### 7.5 Stretch goals (only after everything above)
- Word-level attention vs LLM surprisal (e.g. GPT-2). Only meaningful if validated accuracy is much better than expected.
- Drift correction using known-position events (e.g. re-center dot every few pages).

---

## 8. Milestones and acceptance criteria

**Build order: app first, dataset evaluation last.** Work one milestone at a time. At the end of each, run the checks, print the numbers, and wait for my go-ahead. Do not request the dataset (M5) or the L2CS weights (M3) before the milestone that needs them.

**M0 — Environment and sanity (half day)**
- `scripts/check_env.py` prints Python version, architecture, mediapipe/torch/opencv versions, MPS availability, and whether the camera opens.
- `scripts/debug_landmarks.py` shows a live window with iris centers, eye corners and lid points labeled by index so landmark indices can be verified visually.
- Accept: I can see correct iris/corner labeling; blink gives `valid=False`.

**M1 — Live calibration and gaze cursor (Method A)**
- Landmark feature extractor, FastAPI server + fullscreen page, 9-point calibration, validation dots, live gaze cursor with One Euro smoothing, session log.
- Accept: measured live `sigma_px` shown on screen and logged; the whole flow runs from one command; about 30 fps without visible lag; calibration math has a unit test (known linear map is recovered exactly).

**M2 — Reading page, fixations, heatmap**
- Paged reading UI, line-box capture, I-DT, soft line mapping, heatmap and scanpath views, JSON/PNG export, automatic line vs paragraph granularity.
- Accept: replaying a logged session offline reproduces the same heatmap; unit tests for I-DT and line mapping on hand-made inputs. **This is the first complete, demoable product.**

**M3 — Method B (L2CS-Net) as an alternative backend**
- Wrapper behind the same interface as Method A, selectable with a config flag. This is when you may ask me for the L2CS weights.
- Accept: both backends run in the live app; live `sigma_px` for A vs B on the same person and setup is reported side by side, with a short note on which is better and why. If the weights are unavailable, tell me and stop.

**M4 — Simulation experiments and design guidelines (no dataset needed)**
- Everything in 7.3, regenerated by one command, using the `sigma_px` values measured in M1/M3.
- Accept: figures of correlation vs noise sigma and line pitch, plus the minimum line pitch result.

**M5 — Offline evaluation on MPIIFaceGaze (secondary, run last)**
- Ask me to download the dataset now. Loader with unit tests (annotation parsing, px<->mm conversion, per-person splits, no person in two folds). Feature extraction cached to `data/processed/`.
- LOPO + simulated k-shot calibration for Method A (affine and poly2) and Method B, exactly as in 7.1.
- Accept: reproducible results table (mm and % screen height) for k in {1,4,9,16,25}, per-participant breakdown, fraction of excluded frames, A vs B figures. A test asserts no calibration frame appears in its own evaluation set.
- If time is short, cut scope in this order: Method B, then k values, then participants (use at least 5). Do not cut the no-leakage test.

**M6 — Human check, polish, write-up support**
- Section 7.4 run, README with exact commands, `results/` populated, short demo script (2 minutes) for the course presentation.
- Accept: a fresh clone can reproduce all tables and figures from 7.1 and 7.3 with documented commands.

Course deadline and presentation date: **TODO — fill in**. Plan backwards from it and reserve a fixed block for M5 so it does not get squeezed out. M1–M2 must be demoable well before the deadline.

---

## 9. Testing and quality bar
- `pytest` for: loader, calibration math (known linear map must be recovered exactly), k-shot selection (spread and no leakage), I-DT on synthetic traces, line mapping on synthetic layouts, filter behavior on gaps.
- Type hints on public functions; docstrings with units (px, mm, seconds, degrees) for every quantity.
- Any number shown in README or the report must come from a script in `eval/` that writes it to `results/`. No hand-typed numbers.
- Log at INFO for milestones and WARNING for dropped frames, never print inside tight loops.
- Before finishing each milestone: run tests, run the milestone's acceptance check, summarize what changed, list known limitations, and list what you could not verify.

---

## 10. What not to do
- Do not claim word-level accuracy, "eye tracking" equivalence, or real-time performance without measurement.
- Do not tune hyperparameters on the held-out person or on the same frames used to report results.
- Do not save face video or images by default. Do not commit data, weights, sessions or personal info.
- Do not add cloud services, accounts, telemetry or external API calls.
- Do not start Method C, word-level work or surprisal analysis before M5 is accepted.
- Do not block app development (M1–M4) on the dataset. Nothing in the live pipeline may depend on MPIIFaceGaze being present.
- Do not rewrite finished modules while working on a later milestone unless a test proves a bug; tell me first.

---

## 11. Commands (create these and keep them current)
```
make setup          # venv + pip install -r requirements.txt
make check          # python scripts/check_env.py
make test           # pytest -q
make eval-lopo      # M5 tables and figures (needs the dataset)
make sim            # M4 experiments
make app            # start server, open http://localhost:8000 in a fullscreen browser
make results        # everything under results/
```
