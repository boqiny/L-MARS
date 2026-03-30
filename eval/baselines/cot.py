"""Condition 2 — Chain-of-Thought baseline for Bar Exam QA."""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from tqdm import tqdm

from eval.api_utils import call_gpt, load_completed_ids, parse_answer_letter, save_result
from eval.config import MAX_TOKENS_COT, MODEL
from eval.prompts import BAREXAM_COT


def run_cot_barexam(rows: List[dict], output_path: Path, workers: int = 5) -> List[dict]:
    """Run chain-of-thought on Bar Exam QA rows with parallel API calls.

    Already-completed question IDs (from a prior run) are skipped.
    Results are appended to ``output_path`` as they complete.
    """
    completed = load_completed_ids(output_path)
    pending = [r for r in rows if (r["example_id"] or r["idx"]) not in completed]

    if not pending:
        return []

    def _process(row: dict) -> dict:
        row_id = row["example_id"] or row["idx"]
        prompt_text = BAREXAM_COT.format(
            prompt=row["prompt"],
            question=row["question"],
            choice_a=row["choice_a"],
            choice_b=row["choice_b"],
            choice_c=row["choice_c"],
            choice_d=row["choice_d"],
        )
        t0 = time.perf_counter()
        try:
            raw = call_gpt(prompt_text, max_tokens=MAX_TOKENS_COT)
        except Exception as e:
            raw = f"API_ERROR: {e}"
            tqdm.write(f"ERROR [id={row_id}]: GPT call failed: {e}", file=sys.stderr)
        latency = time.perf_counter() - t0

        pred = parse_answer_letter(raw, row_id=row_id)
        gold = row["answer"].strip().upper()
        result = {
            "id": row_id,
            "subject": row["subject"],
            "question": row["question"],
            "gold": gold,
            "pred": pred,
            "correct": pred == gold,
            "raw_response": raw,
            "condition": "cot",
            "model": MODEL,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_s": round(latency, 3),
        }
        save_result(output_path, result)
        return result

    results: List[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process, row): row for row in pending}
        for future in tqdm(as_completed(futures), total=len(futures), desc="cot/barexam"):
            try:
                results.append(future.result())
            except Exception as e:
                tqdm.write(f"ERROR: unexpected thread failure: {e}", file=sys.stderr)

    return results
