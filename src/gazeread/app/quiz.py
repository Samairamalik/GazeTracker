"""Comprehension questions for the human check (spec 7.4).

Each passage `texts/<name>.txt` may have `texts/<name>.questions.json`:
{"passage": name, "questions": [{"id", "para_id", "text", "options": [4 strings], "answer": index}]}.
`para_id` is the paragraph (0-based, blank-line separated) the question is about. The browser only
receives the questions without the answer key; answers are scored here and logged as `answer`.
"""
import json
from pathlib import Path


def load_questions(texts_dir: str | Path, name: str) -> list[dict]:
    """All questions for passage `name`, or [] if it has none."""
    p = Path(texts_dir) / f"{name}.questions.json"
    return json.loads(p.read_text(encoding="utf-8"))["questions"] if p.exists() else []


def public_questions(questions: list[dict]) -> list[dict]:
    """Questions as sent to the page: no answer key."""
    return [{k: q[k] for k in ("id", "text", "options")} for q in questions]


def score(questions: list[dict], question_id: str, response: int) -> tuple[int, bool]:
    """(para_id, correct) for one answer; KeyError for an unknown question."""
    q = next(q for q in questions if q["id"] == question_id)
    return int(q["para_id"]), int(response) == int(q["answer"])
