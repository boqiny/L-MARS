#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys

from .workflow import create_workflow


def main() -> int:
    parser = argparse.ArgumentParser(description="L-MARS single-turn CLI")
    parser.add_argument("query", nargs="?", help="Legal question")
    parser.add_argument("--model", default="openai:gpt-4o-mini", help="Model ID")
    parser.add_argument("--example-id", default="cli", help="Cache key id")
    parser.add_argument("--use-cache", default="true", choices=["true", "false"])
    parser.add_argument("--cache-dir", default="eval/cache")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    if not args.query:
        parser.print_help()
        return 0

    workflow = create_workflow(mode="simple", llm_model=args.model)
    result = workflow.run(
        query=args.query,
        example_id=args.example_id,
        use_cache=args.use_cache == "true",
        cache_dir=args.cache_dir,
        seed=args.seed,
    )
    print(result["final_answer"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
