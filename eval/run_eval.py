#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import json
from pathlib import Path

from eval.compute_metrics import compute_f1, run_llm_as_judge


def main() -> int:
    parser = argparse.ArgumentParser(description="Run F1 + LLM-as-judge evaluation")
    parser.add_argument("--preds", required=True)
    parser.add_argument("--judge-sample", type=int, default=50)
    parser.add_argument("--llm_model", default="mock")
    parser.add_argument("--metrics-out", default="eval/results/metrics.json")
    parser.add_argument("--judge-out", default="eval/results/judge.jsonl")
    args = parser.parse_args()

    pred_rows = []
    with Path(args.preds).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                pred_rows.append(json.loads(line))

    preds = [row.get("pred", "") for row in pred_rows]
    golds = [row.get("gold", "") for row in pred_rows if row.get("gold") is not None]

    if len(golds) != len(preds):
        raise ValueError("All prediction rows must include gold for F1 computation.")

    f1 = compute_f1(preds, golds)
    judged = run_llm_as_judge(args.preds, args.judge_sample, args.llm_model)

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "num_examples": len(pred_rows),
        "f1_micro": f1,
        "judge_samples": len(judged),
        "judge_model": args.llm_model,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    judge_path = Path(args.judge_out)
    judge_path.parent.mkdir(parents=True, exist_ok=True)
    with judge_path.open("w", encoding="utf-8") as f:
        for row in judged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
