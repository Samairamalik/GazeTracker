# GazeRead: webcam gaze-based attention heatmaps for reading

Course project (Principles of Computer Vision). An ordinary laptop webcam estimates where on the
screen a person is looking while reading, and the app turns that into a heatmap of which
paragraphs (and, only when validated, which lines) received attention.

**Honest framing.** Webcam gaze is coarse: errors are a few degrees of visual angle, i.e.
several centimetres on screen, often more than a line of text. The app therefore measures its
own calibration error and picks **paragraph-level** output unless line-level is supported by that
measurement. It is not an eye tracker, and it does not resolve words.

The full specification is [`SPEC.md`](SPEC.md).

## What is here

| Part | What it does | Code |
|---|---|---|
| Live app | Fullscreen calibration (9 or 16 dots, shuffled), held-out validation dots → σ in px, smoothed gaze cursor, paged reading, fixations, line/paragraph heatmap + scanpath, JSON/PNG export | `src/gazeread/app/`, `make app` |
| Method A | MediaPipe landmarks → iris position, eye openness, head pose → ridge calibration map | `gaze/backend_landmark.py`, `gaze/calibration.py` |
| Method B | L2CS-Net (ResNet50, Gaze360 weights) on the MediaPipe face crop → gaze angles → the same calibration map | `gaze/backend_l2cs.py`, `gaze/l2cs_model.py` |
| Replay | Recomputes a session's heatmap from its log alone and checks it equals the live result | `make replay` |
| Simulation (spec 7.3) | Synthetic reading + webcam-like noise through the app's own analysis → line/paragraph fidelity vs error, line pitch, paragraph height | `make sim` |
| Dataset evaluation (spec 7.1) | MPIIFaceGaze, leave-one-person-out, simulated k-point calibration, A vs B | `make eval-lopo` |

## Privacy

- No video or face images are saved: session logs hold landmark-derived features, gaze
  coordinates and events only (`sessions/<id>/events.jsonl`, gitignored).
- The app prints and shows a one-line consent notice at start.
- No cloud services, accounts or telemetry. The only network downloads are the model files and
  the dataset listed below, each fetched once and verified.

## Setup (macOS, Apple Silicon, Python 3.11)

```bash
make setup        # .venv with uv (Python 3.11) + pinned requirements + MediaPipe face model
make weights      # Method B weights (95.8 MB) from Hugging Face, sha256-checked
make check        # versions, MPS, models load, camera opens
make test         # unit tests
```

Camera access on macOS belongs to the app that runs Python: run `make app` from **Terminal**
and allow camera access there (System Settings → Privacy & Security → Camera).

### External files (never committed)

| File | Used by | How to get it |
|---|---|---|
| `models/face_landmarker.task` | everything live | `make setup` (official MediaPipe model URL) |
| `models/l2cs_gaze360_resnet50.safetensors` | Method B, compare mode, `eval-lopo` | `make weights`; source, revision and sha256 in [`CITATIONS.md`](CITATIONS.md). The code is MIT; the weights follow the Gaze360 research terms (non-commercial). |
| `data/raw/MPIIFaceGaze/` | `extract-mpii`, `eval-lopo` | Download https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIFaceGaze.zip (942,343,290 bytes, CC BY-NC-SA 4.0) and unzip it into `data/raw/` so that `data/raw/MPIIFaceGaze/p00 … p14` exist. Do **not** download the 22 GB normalized version. |

## Running the app

```bash
make app                          # Method A, 9 dots, opens the browser
make app N_POINTS=16              # 16 calibration dots
make app FEATURES=iris            # Method A features: iris | iris_pose (default) | iris_open
make app BACKEND=l2cs             # Method B drives the cursor
make app COMPARE=1                # both methods every frame; one calibration scores A, B, B+pose
make app SOURCE=synthetic         # no camera: fake gaze for testing (never an accuracy result)
```

In the page: **Enter** fullscreen and calibrate → **Space** to read (Space = next page) →
heatmap views **L** line, **P** paragraph, **S** scanpath (coarse), **F** fixations/samples,
**←/→** pages. **C** recalibrates, **H** hides the HUD. Exports land in `sessions/<id>/`.

```bash
make app TEXT=alice_passage1           # another passage (src/gazeread/app/static/texts/<name>.txt);
                                       # passages with <name>.questions.json end with a quiz (keys 1–4)
make app TEXT=alice_passage2 SKIM=1    # mark the session as a deliberately skimmed reading
make replay SESSION=sessions/<id>                      # offline heatmap; checks equality with the live one
make compare SESSIONS="sessions/<id1> sessions/<id2>"  # A vs B vs B+pose from compare-mode sessions
```

## Reproducing the results

```bash
make sim          # spec 7.3: ~90 s → results/sim_summary.md, results/tables/sim_*.csv, results/figures/sim_*.png
make eval-lopo    # spec 7.1: feature extraction (~20 min, cached in data/processed/) + LOPO (~5 min)
                  #           → results/m5_summary.md, results/tables/m5_*.csv, results/figures/m5_*.png
make results      # both
```

Everything is seeded; reruns on the same machine give the same tables (see the M6 fresh-clone
check below).

**One input cannot be regenerated from a fresh clone:** `results/tables/measured_sigma.csv` and
`measured_sigma_summary.csv` (the live calibration errors that shape the simulation's noise) come
from the authors' own session logs, which are private and not committed. They are refreshed with
`make measured-sigma` only on the machine that holds those sessions. `make sim` reads the
committed copies, so the simulation itself is reproducible anywhere.

### Results

- **Simulation and design guidelines (spec 7.3):** [`results/sim_summary.md`](results/sim_summary.md)
- **MPIIFaceGaze, leave-one-person-out (spec 7.1):** [`results/m5_summary.md`](results/m5_summary.md)
- **Human check (spec 7.4), single reader, illustrative only:** protocol in
  [`docs/human_check.md`](docs/human_check.md); results in [`results/m6_summary.md`](results/m6_summary.md)
  (`make human-check`). Run once with one reader; one page of passage 3 was skipped by an
  accidental double key press and its paragraphs are excluded (see the summary).
- Tables: [`results/tables/`](results/tables/); figures: [`results/figures/`](results/figures/)

## Known limitations

- **Gaze error is large and session-dependent.** Live calibrations of the same person vary a lot
  between sessions, and most of the error is a slowly drifting offset rather than jitter (see
  `results/tables/measured_sigma.csv` and `measured_sigma_summary.csv`), so recalibrating before
  each reading matters.
- **Line-level heatmaps are not supported at our measured error** on a page that fits the
  screen; the simulation summary gives the threshold. Paragraph-level output is the honest
  default, with paragraphs of about six lines.
- The **live A vs B comparison** is a handful of paired calibrations from one person, so it is
  anecdotal; the dataset evaluation is the stronger evidence.
- Fixations from I-DT at ~30 fps are coarse (a short fixation is 3–4 samples), and saccades
  between words are smaller than the noise.
- The **simulation** uses a simplified reading model (character grid, no word lengths or
  difficulty) and noise shaped by one person's calibrations.
- **Method B** uses third-party re-hosted official weights (provenance and hash in
  `CITATIONS.md`) with our own MediaPipe face crop instead of the original RetinaFace crop.
- The reading column has a fixed width of 60 characters; on screens narrower than about 1100 CSS px
  lines would overflow.

## Repository layout

```
configs/       default.yaml (app), sim.yaml (7.3), eval_lopo.yaml (7.1)
src/gazeread/  types.py · features/ · gaze/ · analysis/ · app/ · sim/ · eval/ · data/
scripts/       check_env.py, debug_landmarks.py, replay_session.py, compare_backends.py
results/       committed tables, figures and generated summaries
tests/         pytest suite (make test)
docs/          demo script (and the human-check protocol, if run)
```

Citations and licences: [`CITATIONS.md`](CITATIONS.md).
