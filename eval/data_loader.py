"""Dataset loading utilities for the L-MARS evaluation pipeline."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List

_INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


def load_barexam_labeled(filepath: str | Path) -> List[dict]:
    """Load only subject-labeled Bar Exam QA rows (n=594).

    Filters to rows where the ``subject`` column is non-empty.
    Expected columns: idx, dataset, example_id, prompt_id, source, subject,
    question_number, prompt, question, choice_a, choice_b, choice_c,
    choice_d, answer, gold_passage, gold_idx
    """
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r["subject"].strip()]
    return rows


def load_legalsearchqa(filepath: str | Path) -> List[dict]:
    """Load LegalSearchQA dataset and normalise to the barexam row schema.

    The JSON file contains a list of objects with keys: id, question,
    choices (list[str]), gold (0-indexed int), domain, category, difficulty,
    source_url, source_name, date_verified, rationale.

    Returned rows have the same keys that all downstream eval code expects:
    idx, example_id, subject, prompt, question, choice_a .. choice_d, answer.
    """
    with open(filepath, encoding="utf-8") as f:
        raw_rows = json.load(f)

    rows: List[dict] = []
    for r in raw_rows:
        choices = r["choices"]
        rows.append({
            "idx": r["id"],
            "example_id": r["id"],
            "subject": r.get("domain", ""),
            "prompt": "",  # legal_gaia has no separate prompt/context field
            "question": r["question"],
            "choice_a": choices[0] if len(choices) > 0 else "",
            "choice_b": choices[1] if len(choices) > 1 else "",
            "choice_c": choices[2] if len(choices) > 2 else "",
            "choice_d": choices[3] if len(choices) > 3 else "",
            "answer": _INDEX_TO_LETTER.get(r["gold"], "A"),
        })
    return rows
