#!/usr/bin/env python3
"""Phase 1: 50-question pilot run on Bar Exam QA.

Randomly samples 50 questions (seed=42) from the 594 labeled rows, runs all
4 conditions, prints accuracy per condition, and applies a decision gate
before recommending a full run.

Usage:
    python3 -m eval.run_pilot [--conditions zero_shot cot naive_rag lmars]
                               [--workers W] [--skip-gate]

Pass a subset of --conditions to re-run specific ones only (already-completed
questions are skipped via checkpointing).
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.config import BAREXAM_CSV, LEGALSEARCHQA_JSON, PILOT_DIR, PILOT_N, PILOT_SEED
from eval.data_loader import load_barexam_labeled, load_legalsearchqa
from eval.evaluate import (
    compute_barexam_metrics,
    load_results,
    print_barexam_table,
    summarize_parse_errors,
)

ALL_CONDITIONS = ["zero_shot", "cot", "naive_rag", "lmars"]


def main() -> None:
    parser = argparse.ArgumentParser(description="L-MARS Pilot Evaluation (n=50)")
    parser.add_argument(
        "--dataset",
        choices=["barexam", "legalsearchqa"],
        default="barexam",
        help="Which dataset to evaluate (default: barexam)",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=ALL_CONDITIONS,
        default=ALL_CONDITIONS,
        help="Which conditions to run (default: all four)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        metavar="W",
        help="Parallel API call workers per condition (default: 5)",
    )
    parser.add_argument(
        "--skip-gate",
        action="store_true",
        help="Skip the decision gate check at the end",
    )
    args = parser.parse_args()

    # ── load data ─────────────────────────────────────────────────────────────
    if args.dataset == "legalsearchqa":
        print(f"Loading LegalSearchQA from {LEGALSEARCHQA_JSON} …")
        all_rows = load_legalsearchqa(LEGALSEARCHQA_JSON)
    else:
        print(f"Loading Bar Exam QA from {BAREXAM_CSV} …")
        all_rows = load_barexam_labeled(BAREXAM_CSV)
    print(f"  {len(all_rows)} labeled rows found.")

    n_pilot = min(PILOT_N, len(all_rows))
    random.seed(PILOT_SEED)
    pilot_rows = random.sample(all_rows, n_pilot)
    print(f"  Sampled {len(pilot_rows)} rows for pilot (seed={PILOT_SEED}).\n")

    # ── run conditions ────────────────────────────────────────────────────────
    pilot_dir = PILOT_DIR.parent / args.dataset / "pilot"
    pilot_dir.mkdir(parents=True, exist_ok=True)
    results_map: dict = {}

    for condition in args.conditions:
        output_path = pilot_dir / f"{condition}.jsonl"
        print(f"{'='*60}")
        print(f"Running condition: {condition}  →  {output_path.name}")
        print(f"{'='*60}")

        if condition == "zero_shot":
            from eval.baselines.zero_shot import run_zero_shot_barexam
            run_zero_shot_barexam(pilot_rows, output_path, workers=args.workers)

        elif condition == "cot":
            from eval.baselines.cot import run_cot_barexam
            run_cot_barexam(pilot_rows, output_path, workers=args.workers)

        elif condition == "naive_rag":
            from eval.baselines.naive_rag import run_naive_rag_barexam
            run_naive_rag_barexam(pilot_rows, output_path, workers=args.workers)

        elif condition == "lmars":
            from eval.lmars_eval import run_lmars_barexam
            run_lmars_barexam(pilot_rows, output_path, workers=args.workers)

        # ── print result immediately after this condition finishes ────────────
        if output_path.exists():
            results = load_results(output_path)
            results_map[condition] = results
            m = compute_barexam_metrics(results)
            errs = f"  parse_errors={m['parse_errors']}" if m["parse_errors"] else ""
            print(f"\n  [{condition}] accuracy={m['overall_accuracy']:.1%}  ({m['correct']}/{m['total']}){errs}\n")

    # ── final comparison table + decision gate ────────────────────────────────
    if results_map:
        metrics_map = {c: compute_barexam_metrics(r) for c, r in results_map.items()}

        if len(results_map) > 1:
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            print_barexam_table(metrics_map)
            print("\n--- Parse error summary ---")
            summarize_parse_errors(results_map)

        if not args.skip_gate and "lmars" in metrics_map and "naive_rag" in metrics_map:
            lmars_acc = metrics_map["lmars"]["overall_accuracy"]
            rag_acc = metrics_map["naive_rag"]["overall_accuracy"]
            delta = lmars_acc - rag_acc
            print(f"\n{'='*60}")
            print(f"DECISION GATE: L-MARS ({lmars_acc:.1%}) vs Naive RAG ({rag_acc:.1%})")
            print(f"  Delta = {delta:+.1%}")
            if delta >= 0.03:
                print("  ✓ L-MARS beats Naive RAG by ≥3%. Proceed to full run.")
            else:
                print("  ✗ Delta < 3%. STOP and reassess before spending credits on full run.")
                print("    Re-run with --skip-gate to force a full run anyway.")
            print(f"{'='*60}")


if __name__ == "__main__":
    main()
