#!/usr/bin/env python3
"""Evaluation on Bar Exam QA or Legal GAIA datasets.

Usage:
    python3 -m eval.run_full [--dataset barexam|legalsearchqa]
                              [--conditions zero_shot cot naive_rag lmars lmars_multiturn]
                              [--n N] [--workers W]
                              [--gpt4o-reference]

All runs support resume via checkpointing — already-completed question IDs
are detected and skipped.

Options:
    --dataset             Which dataset to evaluate (default: barexam)
    --conditions          Which conditions to run (default: all 6)
    --n N                 Randomly sample N questions (seed=42). Omit to use
                          all rows. Results go to results/n{N}/ so they never
                          collide with full-run checkpoints.
    --workers W           Parallel API call workers per condition (default: 5)
    --gpt4o-reference     Also run GPT-4o zero-shot as an upper-bound reference
                          (uses gpt-4o, billed separately)
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.config import BAREXAM_CSV, FULL_DIR, LEGALSEARCHQA_JSON, PILOT_SEED, RESULTS_DIR
from eval.data_loader import load_barexam_labeled, load_legalsearchqa
from eval.evaluate import (
    compute_barexam_metrics,
    load_results,
    print_barexam_table,
    summarize_parse_errors,
)

ALL_CONDITIONS = ["zero_shot", "cot", "naive_rag", "lmars", "lmars_multiturn", "lmars_tavily"]


def main() -> None:
    parser = argparse.ArgumentParser(description="L-MARS Evaluation (Bar Exam QA / Legal GAIA)")
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
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        metavar="N",
        help="Randomly sample N questions (seed=42). Omit to use all 594.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        metavar="W",
        help="Parallel API call workers per condition (default: 5)",
    )
    parser.add_argument(
        "--gpt4o-reference",
        action="store_true",
        help="Also run GPT-4o zero-shot as an upper-bound reference",
    )
    args = parser.parse_args()

    # ── load & optionally subsample ───────────────────────────────────────────
    if args.dataset == "legalsearchqa":
        print(f"Loading Legal GAIA from {LEGALSEARCHQA_JSON} …")
        all_rows = load_legalsearchqa(LEGALSEARCHQA_JSON)
    else:
        print(f"Loading Bar Exam QA from {BAREXAM_CSV} …")
        all_rows = load_barexam_labeled(BAREXAM_CSV)
    print(f"  {len(all_rows)} labeled rows total.")

    # Use a dataset-specific subdirectory so results never collide
    dataset_dir = RESULTS_DIR / args.dataset

    if args.n is not None:
        if args.n > len(all_rows):
            parser.error(f"--n {args.n} exceeds dataset size ({len(all_rows)})")
        random.seed(PILOT_SEED)
        rows = random.sample(all_rows, args.n)
        out_dir = dataset_dir / f"n{args.n}"
        print(f"  Sampled {args.n} rows (seed={PILOT_SEED}) → results go to {out_dir}/\n")
    else:
        rows = all_rows
        out_dir = dataset_dir / "full"
        print(f"  Using all {len(rows)} rows → results go to {out_dir}/\n")

    out_dir.mkdir(parents=True, exist_ok=True)

    results_map: dict = {}

    for condition in args.conditions:
        output_path = out_dir / f"{condition}.jsonl"
        print(f"{'='*60}")
        print(f"Running condition: {condition}  →  {output_path.name}")
        print(f"{'='*60}")

        if condition == "zero_shot":
            from eval.baselines.zero_shot import run_zero_shot_barexam
            run_zero_shot_barexam(rows, output_path, workers=args.workers)

        elif condition == "cot":
            from eval.baselines.cot import run_cot_barexam
            run_cot_barexam(rows, output_path, workers=args.workers)

        elif condition == "naive_rag":
            from eval.baselines.naive_rag import run_naive_rag_barexam
            run_naive_rag_barexam(rows, output_path, workers=args.workers)

        elif condition == "lmars":
            from eval.lmars_eval import run_lmars_barexam
            run_lmars_barexam(rows, output_path, workers=args.workers)

        elif condition == "lmars_multiturn":
            from eval.lmars_eval import run_lmars_multiturn_barexam
            run_lmars_multiturn_barexam(rows, output_path, workers=args.workers)

        elif condition == "lmars_tavily":
            from eval.lmars_eval import run_lmars_tavily_barexam
            run_lmars_tavily_barexam(rows, output_path, workers=args.workers)

        # ── print result immediately after this condition finishes ────────────
        if output_path.exists():
            results = load_results(output_path)
            results_map[condition] = results
            m = compute_barexam_metrics(results)
            errs = f"  parse_errors={m['parse_errors']}" if m["parse_errors"] else ""
            print(f"\n  [{condition}] accuracy={m['overall_accuracy']:.1%}  ({m['correct']}/{m['total']}){errs}\n")

    # ── final comparison table across all completed conditions ────────────────
    if len(results_map) > 1:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        metrics_map = {c: compute_barexam_metrics(r) for c, r in results_map.items()}
        print_barexam_table(metrics_map)
        print("\n--- Parse error summary ---")
        summarize_parse_errors(results_map)

    # ── optional GPT-4o upper-bound reference ─────────────────────────────────
    if args.gpt4o_reference:
        _run_gpt4o_reference(rows, out_dir)


def _run_gpt4o_reference(rows: list, out_dir: Path) -> None:
    """Zero-shot with GPT-4o as an upper-bound reference."""
    import time
    from datetime import datetime, timezone

    from tqdm import tqdm

    from eval.api_utils import call_gpt, load_completed_ids, parse_answer_letter, save_result
    from eval.config import MAX_TOKENS_ZERO_SHOT
    from eval.prompts import BAREXAM_ZERO_SHOT

    output_path = out_dir / "gpt4o_reference.jsonl"
    completed = load_completed_ids(output_path)

    print(f"\n{'='*60}")
    print("GPT-4o zero-shot reference run")
    print(f"{'='*60}")

    results = []
    for row in tqdm(rows, desc="gpt4o_reference/barexam"):
        row_id = row["example_id"] or row["idx"]
        if row_id in completed:
            continue

        prompt_text = BAREXAM_ZERO_SHOT.format(
            prompt=row["prompt"],
            question=row["question"],
            choice_a=row["choice_a"],
            choice_b=row["choice_b"],
            choice_c=row["choice_c"],
            choice_d=row["choice_d"],
        )

        t0 = time.perf_counter()
        try:
            raw = call_gpt(prompt_text, model="gpt-4o", max_tokens=MAX_TOKENS_ZERO_SHOT)
        except Exception as e:
            raw = f"API_ERROR: {e}"
            tqdm.write(f"ERROR [id={row_id}]: GPT-4o call failed: {e}", file=sys.stderr)
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
            "condition": "gpt4o_reference",
            "model": "gpt-4o",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "latency_s": round(latency, 3),
        }
        save_result(output_path, result)
        results.append(result)

    from eval.evaluate import compute_barexam_metrics, load_results
    all_results = load_results(output_path)
    m = compute_barexam_metrics(all_results)
    print(f"\n  [gpt4o_reference] accuracy={m['overall_accuracy']:.1%}  ({m['correct']}/{m['total']})")


if __name__ == "__main__":
    main()
