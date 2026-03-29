"""Scoring, metrics, and reporting for the L-MARS evaluation pipeline."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


# ── Load results ──────────────────────────────────────────────────────────────

def load_results(filepath: str | Path) -> List[dict]:
    """Load all JSONL result lines from a file."""
    rows: List[dict] = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ── Bar Exam QA metrics ───────────────────────────────────────────────────────

def compute_barexam_metrics(results: List[dict]) -> dict:
    """Compute overall and per-subject accuracy for Bar Exam QA results."""
    subject_correct: dict = defaultdict(int)
    subject_total: dict = defaultdict(int)
    total_correct = 0
    parse_errors = 0

    for r in results:
        if r.get("pred") == "PARSE_ERROR":
            parse_errors += 1
        subject_total[r["subject"]] += 1
        if r.get("correct"):
            subject_correct[r["subject"]] += 1
            total_correct += 1

    n = len(results)
    metrics: dict = {
        "overall_accuracy": total_correct / n if n else 0.0,
        "total": n,
        "correct": total_correct,
        "parse_errors": parse_errors,
        "parse_error_rate": parse_errors / n if n else 0.0,
        "per_subject": {},
    }
    for subj in subject_total:
        tot = subject_total[subj]
        cor = subject_correct[subj]
        metrics["per_subject"][subj] = {
            "accuracy": cor / tot if tot else 0.0,
            "correct": cor,
            "total": tot,
        }
    return metrics


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_barexam_table(metrics_by_condition: Dict[str, dict]) -> None:
    """Print Table 1 (overall) and Table 2 (per-subject) to stdout."""
    conditions = list(metrics_by_condition.keys())
    print("\n=== Table 1: Bar Exam QA Overall Results ===")
    header = f"{'Condition':<24} {'Model':<14} {'Accuracy':>8} {'N':>5}"
    print(header)
    print("-" * len(header))
    for cond, m in metrics_by_condition.items():
        print(
            f"{cond:<24} {'gpt-4o-mini':<14} {m['overall_accuracy']:>7.1%} {m['total']:>5}"
        )

    # collect all subjects
    all_subjects: list[str] = []
    for m in metrics_by_condition.values():
        for s in m["per_subject"]:
            if s not in all_subjects:
                all_subjects.append(s)

    print("\n=== Table 2: Bar Exam QA Per-Subject Breakdown ===")
    col_w = 10
    row_header = f"{'Subject':<14} {'N':>5}"
    for c in conditions:
        row_header += f"  {c[:col_w]:>{col_w}}"
    print(row_header)
    print("-" * len(row_header))
    for subj in sorted(all_subjects):
        n = next(
            (m["per_subject"][subj]["total"] for m in metrics_by_condition.values() if subj in m["per_subject"]),
            0,
        )
        row = f"{subj:<14} {n:>5}"
        for cond in conditions:
            acc = metrics_by_condition[cond]["per_subject"].get(subj, {}).get("accuracy", float("nan"))
            row += f"  {acc:>{col_w}.1%}" if acc == acc else f"  {'N/A':>{col_w}}"
        print(row)


def summarize_parse_errors(results_by_condition: Dict[str, List[dict]]) -> None:
    """Warn if parse-error rate exceeds 5% for any condition."""
    for cond, results in results_by_condition.items():
        n = len(results)
        if n == 0:
            continue
        errs = sum(1 for r in results if r.get("pred") == "PARSE_ERROR")
        rate = errs / n
        tag = " *** HIGH ***" if rate > 0.05 else ""
        print(f"  {cond}: {errs}/{n} parse errors ({rate:.1%}){tag}")
