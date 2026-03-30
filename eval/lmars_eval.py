"""Condition 4 — L-MARS (multi-turn) evaluation on Bar Exam QA.

Wraps the L-MARS workflow (QueryAgent → JudgeAgent loop → SummaryAgent)
using GPT-4o-mini for all agents, ensuring a fair comparison against the
baselines that use the same backbone.

Key design choices
------------------
- **Multi-turn retrieval** (max_turns=3): after each search round a JudgeAgent
  checks whether the retrieved snippets contain specific legal doctrine. If not,
  it proposes a refined query targeting primary sources.  This replaces the old
  single-pass approach that silently accepted blocked / irrelevant results.
- **Blocked-result filtering**: results with 403 errors, empty content, or
  JS-disabled pages are stripped before both judging and synthesis.
- **Separate search / synthesis queries**: Serper query = short focused question
  (no answer choices); synthesis prompt = full MCQ with A/B/C/D.
- ``use_deep_content=False`` keeps latency reasonable.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from tqdm import tqdm

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lmars.workflow import create_workflow  # noqa: E402

from eval.api_utils import load_completed_ids, parse_answer_letter, save_result
from eval.config import LMARS_MODEL_ID, MODEL

_SYSTEM_PROMPT_PATH = _ROOT / "prompts" / "barexam_system_prompt.txt"
_SYSTEM_PROMPT = _SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")

# Single-pass workflow (original "lmars" condition)
_WORKFLOW = create_workflow(
    mode="simple",
    llm_model=LMARS_MODEL_ID,
    use_deep_content=False,
    max_results=5,
    max_turns=1,
    system_prompt=_SYSTEM_PROMPT,
)

# Multi-turn workflow — JudgeAgent retries up to 3 rounds ("lmars_multiturn" condition)
_WORKFLOW_MULTITURN = create_workflow(
    mode="simple",
    llm_model=LMARS_MODEL_ID,
    judge_model=LMARS_MODEL_ID,
    use_deep_content=False,
    max_results=5,
    max_turns=3,
    system_prompt=_SYSTEM_PROMPT,
)

# Tavily-backed workflow — AI-optimised extracted content, no 403 blocking
_WORKFLOW_TAVILY = create_workflow(
    mode="simple",
    llm_model=LMARS_MODEL_ID,
    search_backend="tavily",
    max_results=5,
    max_turns=1,
    system_prompt=_SYSTEM_PROMPT,
)


def _run_lmars(
    rows: List[dict],
    output_path: Path,
    workflow,
    condition_name: str,
    workers: int = 5,
) -> List[dict]:
    """Shared runner for both single-pass and multi-turn L-MARS conditions."""
    completed = load_completed_ids(output_path)
    pending = [r for r in rows if (r["example_id"] or r["idx"]) not in completed]

    if not pending:
        return []

    def _process(row: dict) -> dict:
        row_id = row["example_id"] or row["idx"]

        # Focused Serper query — no answer choices to avoid polluting results
        search_query = f"US law bar exam: {row['question'][:200]}"

        # Full MCQ passed to SummaryAgent for synthesis
        synthesis_query = (
            f"{row['prompt']}\n\n"
            f"{row['question']}\n\n"
            f"A. {row['choice_a']}\n"
            f"B. {row['choice_b']}\n"
            f"C. {row['choice_c']}\n"
            f"D. {row['choice_d']}\n"
        )

        t0 = time.perf_counter()
        raw = ""
        error = None
        result_dict: dict = {}
        try:
            result_dict = workflow.run(
                query=synthesis_query,
                search_query=search_query,
                example_id=row_id,
                use_cache=False,
                cache_dir="eval/cache",
                seed=0,
            )
            raw = result_dict.get("final_answer", "")
            if not raw:
                tqdm.write(
                    f"WARNING [id={row_id}]: L-MARS returned empty final_answer",
                    file=sys.stderr,
                )
        except Exception as e:
            raw = f"LMARS_ERROR: {e}"
            error = str(e)
            tqdm.write(f"ERROR [id={row_id}]: L-MARS pipeline failed: {e}", file=sys.stderr)
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
            "search_query": search_query,
            "turns_used": result_dict.get("turns_used", 1) if not error else 1,
            "judge_log": result_dict.get("judge_log", []) if not error else [],
            "condition": condition_name,
            "model": MODEL,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_s": round(latency, 3),
        }
        if error:
            result["error"] = error
        save_result(output_path, result)
        return result

    results: List[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process, row): row for row in pending}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc=f"{condition_name}/barexam"
        ):
            try:
                results.append(future.result())
            except Exception as e:
                tqdm.write(f"ERROR: unexpected thread failure: {e}", file=sys.stderr)

    return results


def run_lmars_barexam(rows: List[dict], output_path: Path, workers: int = 5) -> List[dict]:
    """Single-pass L-MARS: one search round, no judge."""
    return _run_lmars(rows, output_path, _WORKFLOW, "lmars", workers)


def run_lmars_multiturn_barexam(
    rows: List[dict], output_path: Path, workers: int = 5
) -> List[dict]:
    """Multi-turn L-MARS: up to 3 search rounds with JudgeAgent refinement."""
    return _run_lmars(rows, output_path, _WORKFLOW_MULTITURN, "lmars_multiturn", workers)


def run_lmars_tavily_barexam(
    rows: List[dict], output_path: Path, workers: int = 5
) -> List[dict]:
    """Single-pass L-MARS with Tavily search (AI-optimised extracted content)."""
    return _run_lmars(rows, output_path, _WORKFLOW_TAVILY, "lmars_tavily", workers)
