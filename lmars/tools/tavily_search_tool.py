"""Tavily search tool for L-MARS.

Tavily is designed for AI-agent use: it returns clean, extracted page content
rather than raw HTML snippets, which avoids the 403-blocked / Quizlet-noise
problem we see with Serper.

search_depth="advanced" makes Tavily actually crawl and extract content from
each result page server-side — no client-side scraping needed.
"""
from __future__ import annotations

import os
import sys
import time


def search_tavily(
    query: str,
    max_results: int = 5,
    search_depth: str = "advanced",
    max_retries: int = 5,
) -> str:
    """Search via Tavily and return a formatted string of results.

    Retries up to ``max_retries`` times with exponential backoff on rate-limit
    errors (the dev key has lower QPS limits than production keys).

    Args:
        query:        Search query string.
        max_results:  Maximum number of results to return (default 5).
        search_depth: "basic" (fast, snippets only) or "advanced" (extracts
                      full page content server-side). Default: "advanced".
        max_retries:  Max attempts before giving up (default 5).

    Returns:
        Formatted string with title, URL, and extracted content per result.
        Returns an error string on failure — never raises.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY environment variable not set."

    try:
        from tavily import TavilyClient
    except ImportError:
        return "Error: tavily-python not installed. Run: pip install tavily-python"

    client = TavilyClient(api_key)

    for attempt in range(max_retries):
        try:
            response = client.search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
            )
            break  # success
        except Exception as e:
            err_str = str(e).lower()
            is_rate_limit = any(
                kw in err_str for kw in ("excessive", "rate", "429", "too many", "blocked")
            )
            wait = 2 ** attempt  # 1s, 2s, 4s, 8s, 16s
            if is_rate_limit and attempt < max_retries - 1:
                print(
                    f"Tavily rate limit (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {wait}s…",
                    file=sys.stderr,
                )
                time.sleep(wait)
            else:
                print(f"Tavily API error: {e}", file=sys.stderr)
                return f"Error during Tavily search: {e}"
    else:
        return "Error during Tavily search: max retries exceeded"

    results = response.get("results", [])
    if not results:
        return "No Tavily search results found."

    lines = [f"Found {len(results)} web results:\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r.get('title', '(no title)')}")
        lines.append(f"   URL: {r.get('url', '')}")
        content = r.get("content", "").strip()
        if content:
            lines.append(f"   Content: {content}")
        lines.append("")

    return "\n".join(lines)
