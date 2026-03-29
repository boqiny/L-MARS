#!/usr/bin/env python3
"""Ablation: top-k Serper results for the naive RAG condition.

This is NOT part of the main comparison ladder. It answers:
  "Does giving the model more search results actually help?"

Cache note
----------
Top-10 results are always fetched and persisted on the first call. Running
smaller k values costs zero additional Serper API calls — the script simply
runs the largest k first so the cache is fully populated before smaller k
slices are evaluated.

Output
------
Results are written to ``eval/results/ablation_topk/`` so they never touch
the main condition outputs.

Usage
-----
    python3 -m eval.ablation_topk [--n 50] [--k 1 5 10]

Options
-------
    --n N        Randomly sample N questions (seed=42). Omit to use all 594.
    --k K ...    Which k values to run (default: 1 5 10).
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.config import BAREXAM_CSV, LEGALSEARCHQA_JSON, PILOT_SEED, RESULTS_DIR, SERPER_FETCH_K
from eval.data_loader import load_barexam_labeled, load_legalsearchqa
from eval.evaluate import compute_barexam_metrics, load_results, print_barexam_table

ABLATION_DIR = RESULTS_DIR / "ablation_topk"  # overridden per-dataset below


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation: naive RAG top-k Serper results"
    )
    parser.add_argument(
        "--dataset",
        choices=["barexam", "legalsearchqa"],
        default="barexam",
        help="Which dataset to evaluate (default: barexam)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        metavar="N",
        help="Randomly sample N questions (seed=42). Omit to use all.",
    )
    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=[1, 5, 10],
        metavar="K",
        help=f"Which top-k values to test (default: 1 5 10, max {SERPER_FETCH_K}).",
    )
    args = parser.parse_args()

    for kv in args.k:
        if kv < 1 or kv > SERPER_FETCH_K:
            parser.error(f"--k values must be between 1 and {SERPER_FETCH_K} (got {kv})")

    # ── load data ─────────────────────────────────────────────────────────────
    if args.dataset == "legalsearchqa":
        print(f"Loading LegalSearchQA from {LEGALSEARCHQA_JSON} …")
        all_rows = load_legalsearchqa(LEGALSEARCHQA_JSON)
    else:
        print(f"Loading Bar Exam QA from {BAREXAM_CSV} …")
        all_rows = load_barexam_labeled(BAREXAM_CSV)
    print(f"  {len(all_rows)} labeled rows total.")

    ablation_dir = RESULTS_DIR / args.dataset / "ablation_topk"

    if args.n is not None:
        if args.n > len(all_rows):
            parser.error(f"--n {args.n} exceeds dataset size ({len(all_rows)})")
        random.seed(PILOT_SEED)
        rows = random.sample(all_rows, args.n)
        out_dir = ablation_dir / f"n{args.n}"
        print(f"  Sampled {args.n} rows (seed={PILOT_SEED}) → {out_dir}\n")
    else:
        rows = all_rows
        out_dir = ablation_dir / "full"
        print(f"  Using all {len(rows)} rows → {out_dir}\n")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── run each k (largest first so cache is populated before smaller k) ─────
    k_values = sorted(args.k, reverse=True)

    from eval.baselines.naive_rag import run_naive_rag_barexam
    from eval.evaluate import load_results

    results_map: dict = {}

    for k in k_values:
        label = f"naive_rag_k{k}"
        output_path = out_dir / f"{label}.jsonl"
        print(f"{'='*60}")
        print(f"Running k={k}  →  {output_path.name}")
        if k < max(k_values):
            print("  (Serper results served from local cache — no API calls)")
        print(f"{'='*60}")

        run_naive_rag_barexam(rows, output_path, k=k)

        results = load_results(output_path)
        results_map[label] = results
        m = compute_barexam_metrics(results)
        errs = f"  parse_errors={m['parse_errors']}" if m["parse_errors"] else ""
        print(
            f"\n  [k={k}] accuracy={m['overall_accuracy']:.1%}"
            f"  ({m['correct']}/{m['total']}){errs}\n"
        )

    # ── comparison table ──────────────────────────────────────────────────────
    if len(results_map) > 1:
        print("\n" + "=" * 60)
        print("ABLATION SUMMARY: naive RAG top-k")
        print("=" * 60)
        metrics_map = {label: compute_barexam_metrics(r) for label, r in results_map.items()}
        print_barexam_table(metrics_map)

        # delta between max-k and min-k
        k_sorted = sorted(args.k)
        lo_label = f"naive_rag_k{k_sorted[0]}"
        hi_label = f"naive_rag_k{k_sorted[-1]}"
        if lo_label in metrics_map and hi_label in metrics_map:
            delta = (
                metrics_map[hi_label]["overall_accuracy"]
                - metrics_map[lo_label]["overall_accuracy"]
            )
            print(
                f"\nDelta k={k_sorted[-1]} vs k={k_sorted[0]}: {delta:+.1%}"
                f"  ({'more results help' if delta > 0 else 'more results do not help'})"
            )


if __name__ == "__main__":
    main()
