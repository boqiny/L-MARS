#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from baselines.qwen_runner import run_qwen3
from lmars.pipeline_runner import run_single_turn_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run open-source baseline on LEXam")
    parser.add_argument("--model", default="qwen3")
    parser.add_argument("--dataset", default="leexam", choices=["leexam", "lexam"])
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--retriever-config", default="config/retriever.yaml")
    parser.add_argument("--use-cache", default="true", choices=["true", "false"])
    parser.add_argument("--cache-dir", default="eval/cache")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.model.lower() == "qwen3":
        run_qwen3(
            dataset=args.dataset,
            dataset_path=args.dataset_path,
            output=args.output,
            use_cache=args.use_cache == "true",
            retriever_config=args.retriever_config,
            seed=args.seed,
            cache_dir=args.cache_dir,
        )
    else:
        run_single_turn_pipeline(
            dataset=args.dataset,
            dataset_path=args.dataset_path,
            model=args.model,
            retriever_config=args.retriever_config,
            use_cache=args.use_cache == "true",
            output=args.output,
            seed=args.seed,
            cache_dir=args.cache_dir,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
