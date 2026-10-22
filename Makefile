PY := .venv/bin/python
MODEL_URL := https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task

.PHONY: setup check test debug-landmarks

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
