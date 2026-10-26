# Human check protocol (spec 7.4): single reader, illustrative only

> **Note:** the comprehension quiz used for this check has since been removed from the app. This page documents how the check was run; `make human-check` still re-analyses the sessions recorded at the time (results in `results/m6_summary.md`).

**Purpose.** See whether, for one reader, the paragraph-level attention heatmap and comprehension
answers point the same way: paragraphs that got little gaze time should be the ones whose
questions were missed, and a deliberately skimmed passage should show less dwell and more
mistakes. One reader cannot support general claims; results are labelled accordingly.

**Material.** Three public-domain passages from *Alice's Adventures in Wonderland* (Project
Gutenberg #11; chapters V, VI, VII), each 228–241 words in three paragraphs of six lines (the M4
guideline for paragraph-level heatmaps), two paragraphs per page. Each passage has 4–5
multiple-choice questions; every question is about one paragraph, and its answer is supported by
that paragraph only (checked automatically). The reader does not see the questions before reading
the passage. The browser never receives the answer key; the server scores each answer.

**Consent and privacy.** Webcam is used live for gaze estimation only; no video or images are
saved. Logs contain gaze coordinates, features and answers.

## Procedure (about 15 minutes)

Run each command from **Terminal** in the project folder. Each passage is its own session with a
**fresh calibration** (drift is the largest error source). Sit 50–60 cm away; keep the head still.

1. **Passage 1, read normally**
   ```bash
   make app TEXT=alice_passage1
   ```
   Enter → follow the dots → Space → read at your normal pace (Space for the next page) →
   answer the questions with keys 1–4 → look at the heatmap if you like → close the tab, Ctrl+C.
2. **Passage 2, skim deliberately** (spend roughly 10–15 seconds per page, as if glancing
   through it)
   ```bash
   make app TEXT=alice_passage2 SKIM=1
   ```
3. **Passage 3, read normally**
   ```bash
   make app TEXT=alice_passage3
   ```
4. **Analysis**
   ```bash
   make human-check
   ```
   → `results/m6_summary.md`, `results/tables/m6_passages.csv`, `m6_paragraphs.csv`
   (all camera sessions that contain quiz answers; pass `SESSIONS="sessions/<id> …"` to choose).

If a calibration fails or σ is very large, press **C** before reading to recalibrate. If a
passage goes wrong (interrupted, distracted), rerun that command; then list only the good
sessions with `SESSIONS=…`, and say so in the write-up.
