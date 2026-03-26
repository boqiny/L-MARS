#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import json
from datetime import datetime, timezone
from pathlib import Path

from data.lexam.loader import load_lexam
from lmars.agents import QueryAgent, SearchAgent, to_serializable_search_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate retrieval cache snapshots for reproducibility")
    parser.add_argument("--dataset", required=True, help="Path to LEXam jsonl")
    parser.add_argument("--out", default="eval/cache")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--use-deep-content", action="store_true")
    parser.add_argument("--max-results", type=int, default=5)
    args = parser.parse_args()

    rows = load_lexam(args.dataset)[: args.limit]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    query_agent = QueryAgent()
    search_agent = SearchAgent(use_deep_content=args.use_deep_content, max_results=args.max_results)

    for row in rows:
        query = query_agent.build_query(row["input"]).query
        retrieved = search_agent.search(
            example_id=row["id"],
            query=query,
            use_cache=False,
            cache_dir=args.out,
        )
        payload = {
            "id": row["id"],
            "query": query,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "retrieved": to_serializable_search_results(retrieved),
        }
        (out_dir / f"{row['id']}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Wrote {len(rows)} cache files to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
