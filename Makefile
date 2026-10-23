PY := .venv/bin/python
MODEL_URL := https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
# L2CS-Net Gaze360 weights (safetensors, Py-Feat re-host), pinned revision + sha256 (see CITATIONS.md)
L2CS_URL := https://huggingface.co/py-feat/l2cs/resolve/241076966f03b197773ce75d23adc60299473ccd/l2cs_gaze360_resnet50.safetensors
L2CS_FILE := models/l2cs_gaze360_resnet50.safetensors
L2CS_SHA256 := 75405bdc01f7086b3887280fe50c28a14bc62463a53fa749781e4a3cf98eca0e
# Optional overrides for `make app`, e.g. make app N_POINTS=16 FEATURES=iris
APP_ARGS := $(if $(SOURCE),--source $(SOURCE)) $(if $(N_POINTS),--n-points $(N_POINTS)) $(if $(FEATURES),--features $(FEATURES)) \
            $(if $(BACKEND),--backend $(BACKEND)) $(if $(COMPARE),--compare)

.PHONY: setup weights check test debug-landmarks app replay compare sim results measured-sigma extract-mpii eval-lopo

setup:
	test -d .venv || uv venv --python 3.11 .venv
	uv pip install --python $(PY) -r requirements.txt
	mkdir -p models
	test -f models/face_landmarker.task || curl -fsSL -o models/face_landmarker.task $(MODEL_URL)

# Method B weights (95.8 MB): download if missing, then verify the sha256 (fails on mismatch)
weights:
	mkdir -p models
	test -f $(L2CS_FILE) || curl -fSL -o $(L2CS_FILE) $(L2CS_URL)
	echo "$(L2CS_SHA256)  $(L2CS_FILE)" | shasum -a 256 -c -

check:
	$(PY) scripts/check_env.py

test:
	$(PY) -m pytest -q

debug-landmarks:
	$(PY) scripts/debug_landmarks.py

app:
	PYTHONPATH=src $(PY) -m gazeread.app.server --open $(APP_ARGS)

# Recompute a session's heatmap offline and check it equals the live one: make replay SESSION=sessions/<id>
replay:
	$(PY) scripts/replay_session.py $(SESSION) --check

# A vs B vs B+pose from compare-mode sessions: make compare SESSIONS="sessions/<id1> sessions/<id2>"
compare:
	$(PY) scripts/compare_backends.py $(SESSIONS)

# M4: reading simulation + design guidelines (writes results/tables, results/figures, results/sim_summary.md)
sim:
	PYTHONPATH=src $(PY) -m gazeread.eval.sim_experiments

# Refresh results/tables/measured_sigma*.csv from your local sessions/ (numbers only; needs the session logs)
measured-sigma:
	PYTHONPATH=src $(PY) -m gazeread.eval.measured_sigma

# M5: MPIIFaceGaze (needs data/raw/MPIIFaceGaze, see CITATIONS.md). Extraction is cached per participant.
extract-mpii:
	PYTHONPATH=src $(PY) -m gazeread.eval.extract_mpii

eval-lopo: extract-mpii
	PYTHONPATH=src $(PY) -m gazeread.eval.lopo

results: sim eval-lopo
