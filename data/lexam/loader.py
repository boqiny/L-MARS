from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


REQUIRED_FIELDS = ("id", "input", "gold")


def _resolve_file(path_or_dir: str) -> Path:
    path = Path(path_or_dir)
    if path.is_dir():
        candidate = path / "lexam.jsonl"
        if not candidate.exists():
            raise FileNotFoundError(
                f"LEXam file not found. Expected: {candidate}. "
                "Place your dataset at data/lexam/lexam.jsonl or pass --dataset-path."
            )
        return candidate

    if not path.exists():
        raise FileNotFoundError(f"LEXam path not found: {path}")

    return path


def load_lexam(path_or_dir: str) -> List[Dict[str, str]]:
    """Load LEXam JSONL into a list of {id, input, gold} records."""
    dataset_file = _resolve_file(path_or_dir)
    rows: List[Dict[str, str]] = []

    with dataset_file.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {dataset_file}:{line_no}: {exc}") from exc

            missing = [field for field in REQUIRED_FIELDS if field not in obj]
            if missing:
                raise ValueError(
                    f"Invalid LEXam schema at {dataset_file}:{line_no}. "
                    f"Missing fields: {missing}. Required: {list(REQUIRED_FIELDS)}"
                )

            row = {
                "id": str(obj["id"]),
                "input": str(obj["input"]),
                "gold": str(obj["gold"]),
            }
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows found in LEXam file: {dataset_file}")

    return rows
