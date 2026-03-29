"""Shared API utilities: GPT call wrapper, Serper search, answer parsing, checkpointing."""
from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Set

import requests
from openai import OpenAI

from eval.config import (
    MODEL,
    OPENAI_API_KEY,
    RATE_SLEEP_GPT,
    RATE_SLEEP_SERPER,
    SERPER_API_KEY,
    SERPER_CACHE_DIR,
    SERPER_FETCH_K,
    SERPER_MAX_CHARS,
    SERPER_TOP_K,
    TEMPERATURE,
)

_client = OpenAI(api_key=OPENAI_API_KEY)


# ── OpenAI wrapper ────────────────────────────────────────────────────────────

def call_gpt(
    prompt: str,
    model: str = MODEL,
    temperature: float = TEMPERATURE,
    max_tokens: int = 1024,
) -> str:
    """Single API call with exponential-backoff retry (3 attempts).

    Prints a warning to stderr on each failed attempt before retrying.
    Raises on the third failure — callers decide whether to skip or abort.
    """
    for attempt in range(3):
        try:
            response = _client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            time.sleep(RATE_SLEEP_GPT)
            return response.choices[0].message.content.strip()
        except Exception as e:
            wait = 2 ** attempt
            if attempt < 2:
                print(
                    f"WARNING: GPT call attempt {attempt + 1}/3 failed ({type(e).__name__}: {e})"
                    f" — retrying in {wait}s",
                    file=sys.stderr,
                )
                time.sleep(wait)
            else:
                print(
                    f"ERROR: GPT call failed after 3 attempts ({type(e).__name__}: {e})",
                    file=sys.stderr,
                )
                raise


# ── Serper search with local cache ────────────────────────────────────────────

def _serper_cache_path(query: str) -> Path:
    """Return the cache file path for a given query (keyed by MD5 hash)."""
    key = hashlib.md5(query.encode("utf-8")).hexdigest()
    return SERPER_CACHE_DIR / f"{key}.json"


def serper_search(query: str, k: int = SERPER_TOP_K) -> str:
    """Fetch top-SERPER_FETCH_K results from Serper, cache locally, return top-k.

    Cache behaviour:
    - Always fetches and stores SERPER_FETCH_K (10) results on the first call.
    - If an existing cache entry has fewer results than the requested k, the
      cache is transparently refreshed (re-fetched) at SERPER_FETCH_K.
    - Subsequent calls with k ≤ cached count cost zero extra API calls.
    - Cache files live in ``eval/cache/serper/{md5(query)}.json``.

    Args:
        query: Search query string.
        k:     How many cached results to include in the returned string
               (default SERPER_TOP_K=5). Pass k=1 or k=10 for ablations.
    """
    cache_path = _serper_cache_path(query)
    results: List[dict] = []

    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as f:
            cached = json.load(f)
        if len(cached.get("results", [])) >= k:
            results = cached["results"]

    if not results:
        # Fetch top-SERPER_FETCH_K and (re-)persist
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={
                "X-API-KEY": SERPER_API_KEY,
                "Content-Type": "application/json",
            },
            json={"q": query, "num": SERPER_FETCH_K},
            timeout=20,
        )
        resp.raise_for_status()
        organic = resp.json().get("organic", [])
        results = [
            {
                "title": r.get("title", ""),
                "snippet": r.get("snippet", ""),
                "url": r.get("link", ""),
            }
            for r in organic
        ]
        payload = {
            "query": query,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "fetched_k": SERPER_FETCH_K,
            "results": results,
        }
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        time.sleep(RATE_SLEEP_SERPER)

    # Slice to top-k and format
    selected = results[:k]
    snippets: List[str] = []
    total_chars = 0
    for r in selected:
        snippet = r.get("snippet", "")
        if total_chars + len(snippet) > SERPER_MAX_CHARS:
            break
        snippets.append(f"[{r['title']}]: {snippet}")
        total_chars += len(snippet)
    return "\n".join(snippets)


# ── Answer parsing ────────────────────────────────────────────────────────────

def parse_answer_letter(response: str, row_id: str = "") -> str:
    """Extract a single A/B/C/D letter from a model response.

    Tries three increasingly permissive patterns. Prints a warning to stderr
    and returns ``"PARSE_ERROR"`` if none match.
    """
    # "ANSWER: X" (CoT format)
    m = re.search(r"ANSWER:\s*([A-D])", response, re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Standalone letter at the start of the stripped response
    m = re.search(r"^([A-D])[\.\)\s]", response.strip(), re.IGNORECASE)
    if m:
        return m.group(1).upper()

    # Any isolated A–D word
    m = re.search(r"\b([A-D])\b", response.strip())
    if m:
        return m.group(1).upper()

    id_tag = f" [id={row_id}]" if row_id else ""
    print(
        f"WARNING{id_tag}: PARSE_ERROR — could not extract A/B/C/D from response: {response!r:.120}",
        file=sys.stderr,
    )
    return "PARSE_ERROR"


# ── Checkpointing ─────────────────────────────────────────────────────────────

_SAVE_LOCK = threading.Lock()


def save_result(filepath: str | Path, result: dict) -> None:
    """Append one result dict as a JSONL line (thread-safe)."""
    with _SAVE_LOCK:
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")


def load_completed_ids(filepath: str | Path) -> Set[str]:
    """Return the set of ``id`` values already written to a JSONL file.

    Skips lines that are not valid JSON (e.g. truncated/corrupted from a
    mid-write interrupt) and prints a warning to stderr so they are visible.
    """
    completed: Set[str] = set()
    try:
        with open(filepath, encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    completed.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    import sys
                    print(
                        f"WARNING: skipping corrupt line {lineno} in {filepath}",
                        file=sys.stderr,
                    )
    except FileNotFoundError:
        pass
    return completed
