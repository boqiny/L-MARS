"""Condition 3 — Naive RAG baseline for Bar Exam QA.

Two retrieval modes:
- Basic  (use_deep_content=False): title + snippet only (~100-200 chars/result).
  Uses our local Serper cache; subsequent runs with same query cost zero API calls.
- Enhanced (use_deep_content=True): scrapes full webpage content (~2000 chars/result).
  No local cache — makes live HTTP requests to each URL on every run.
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from tqdm import tqdm

from eval.api_utils import call_gpt, load_completed_ids, parse_answer_letter, save_result, serper_search
from eval.config import MAX_TOKENS_RAG, MODEL, SERPER_TOP_K
from eval.prompts import BAREXAM_RAG


def run_naive_rag_barexam(
    rows: List[dict],
    output_path: Path,
    k: int = SERPER_TOP_K,
    workers: int = 5,
    use_deep_content: bool = False,
) -> List[dict]:
    """Run naive RAG on Bar Exam QA rows with parallel API calls.

    Search query: ``"US law: " + question[:200]``

    Args:
        k:                How many Serper results to include in the prompt.
        workers:          Thread-pool size for concurrent API calls.
        use_deep_content: If True, scrape full webpage content via
                          ``search_serper_with_content`` instead of using
                          snippet-only results from the local cache.
    """
    completed = load_completed_ids(output_path)
    pending = [r for r in rows if (r["example_id"] or r["idx"]) not in completed]

    if not pending:
        return []

    # Import enhanced search only when needed
    if use_deep_content:
        from lmars.tools.serper_search_tool import search_serper_with_content as _deep_search
    else:
        _deep_search = None

    def _retrieve(search_query: str) -> str:
        if use_deep_content:
            return _deep_search(search_query, max_results=k)  # type: ignore[misc]
        return serper_search(search_query, k=k)

    condition_name = "naive_rag_deep" if use_deep_content else "naive_rag"
    desc = f"{condition_name}/barexam(k={k})"

    def _process(row: dict) -> dict:
        row_id = row["example_id"] or row["idx"]
        search_query = "US law: " + row["question"][:200]

        t0 = time.perf_counter()
        try:
            snippets = _retrieve(search_query)
        except Exception as e:
            snippets = f"SEARCH_ERROR: {e}"
            tqdm.write(f"ERROR [id={row_id}]: Serper search failed: {e}", file=sys.stderr)

        prompt_text = BAREXAM_RAG.format(
            retrieved_snippets=snippets,
            prompt=row["prompt"],
            question=row["question"],
            choice_a=row["choice_a"],
            choice_b=row["choice_b"],
            choice_c=row["choice_c"],
            choice_d=row["choice_d"],
        )
        try:
            raw = call_gpt(prompt_text, max_tokens=MAX_TOKENS_RAG)
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
            "retrieved_snippets": snippets,
            "search_query": search_query,
            "serper_k": k,
            "use_deep_content": use_deep_content,
            "condition": condition_name,
            "model": MODEL,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_s": round(latency, 3),
        }
        save_result(output_path, result)
        return result

    results: List[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process, row): row for row in pending}
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            try:
                results.append(future.result())
            except Exception as e:
                tqdm.write(f"ERROR: unexpected thread failure: {e}", file=sys.stderr)

    return results
