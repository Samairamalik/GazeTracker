PY := .venv/bin/python
MODEL_URL := https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
# Optional overrides for `make app`, e.g. make app N_POINTS=16 FEATURES=iris
APP_ARGS := $(if $(SOURCE),--source $(SOURCE)) $(if $(N_POINTS),--n-points $(N_POINTS)) $(if $(FEATURES),--features $(FEATURES))

.PHONY: setup check test debug-landmarks app

setup:
	test -d .venv || uv venv --python 3.11 .venv
	uv pip install --python $(PY) -r requirements.txt
	mkdir -p models
	test -f models/face_landmarker.task || curl -fsSL -o models/face_landmarker.task $(MODEL_URL)

check:
	$(PY) scripts/check_env.py

test:
	$(PY) -m pytest -q

debug-landmarks:
	$(PY) scripts/debug_landmarks.py

app:
	PYTHONPATH=src $(PY) -m gazeread.app.server --open $(APP_ARGS)
