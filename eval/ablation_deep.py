#!/usr/bin/env python3
"""Ablation: basic snippet search vs enhanced (full-page) search for naive RAG.

This is NOT part of the main comparison ladder. It answers:
  "Does scraping full webpage content help over just using Serper snippets?"

Retrieval modes
---------------
- basic: title + snippet only from our local Serper cache (~100-200 chars/result).
  Free after the first run — served entirely from local cache.
- deep:  scrapes the actual webpage for each URL (~2000 chars/result).
  Makes live HTTP requests every run; no local cache.

Output
------
Results are written to ``eval/results/ablation_deep/`` to avoid touching
main condition outputs.

Usage
-----
    python3 -m eval.ablation_deep [--n 50] [--k 5] [--workers 3]

Options
-------
    --n N        Randomly sample N questions (seed=42). Omit to use all 594.
    --k K        Number of Serper results to use (default: 5).
    --workers W  Thread-pool size (default: 3 — deep search scrapes URLs so
                 be conservative to avoid rate-limiting).
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

ABLATION_DIR = RESULTS_DIR / "ablation_deep"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation: basic (snippet) vs enhanced (deep) search for naive RAG"
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
        type=int,
        default=5,
        metavar="K",
        help=f"Number of Serper results to use (default: 5, max {SERPER_FETCH_K}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        metavar="W",
        help="Thread-pool size (default: 3; deep search scrapes URLs so keep low).",
    )
    args = parser.parse_args()

    if args.k < 1 or args.k > SERPER_FETCH_K:
        parser.error(f"--k must be between 1 and {SERPER_FETCH_K} (got {args.k})")

    # ── load data ──────────────────────────────────────────────────────────────
    if args.dataset == "legalsearchqa":
        print(f"Loading LegalSearchQA from {LEGALSEARCHQA_JSON} …")
        all_rows = load_legalsearchqa(LEGALSEARCHQA_JSON)
    else:
        print(f"Loading Bar Exam QA from {BAREXAM_CSV} …")
        all_rows = load_barexam_labeled(BAREXAM_CSV)
    print(f"  {len(all_rows)} labeled rows total.")

    ablation_dir = RESULTS_DIR / args.dataset / "ablation_deep"

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

    from eval.baselines.naive_rag import run_naive_rag_barexam

    conditions = [
        ("naive_rag_basic", False),
        ("naive_rag_deep",  True),
    ]

    results_map: dict = {}

    for label, use_deep in conditions:
        output_path = out_dir / f"{label}_k{args.k}.jsonl"
        print(f"{'='*60}")
        print(f"Running: {label}  k={args.k}  →  {output_path.name}")
        if use_deep:
            print("  (scraping full webpage content — live HTTP requests, slower)")
        else:
            print("  (snippet-only — served from local cache if available)")
        print(f"{'='*60}")

        run_naive_rag_barexam(
            rows,
            output_path,
            k=args.k,
            workers=args.workers,
            use_deep_content=use_deep,
        )

        results = load_results(output_path)
        results_map[label] = results
        m = compute_barexam_metrics(results)
        errs = f"  parse_errors={m['parse_errors']}" if m["parse_errors"] else ""
        print(
            f"\n  [{label}] accuracy={m['overall_accuracy']:.1%}"
            f"  ({m['correct']}/{m['total']}){errs}\n"
        )

    # ── comparison table ───────────────────────────────────────────────────────
    if len(results_map) == 2:
        print("\n" + "=" * 60)
        print(f"ABLATION SUMMARY: basic vs deep search  (k={args.k})")
        print("=" * 60)
        metrics_map = {label: compute_barexam_metrics(r) for label, r in results_map.items()}
        print_barexam_table(metrics_map)

        delta = (
            metrics_map["naive_rag_deep"]["overall_accuracy"]
            - metrics_map["naive_rag_basic"]["overall_accuracy"]
        )
        print(
            f"\nDelta (deep − basic): {delta:+.1%}"
            f"  ({'deep search helps' if delta > 0 else 'deep search does not help'})"
        )


if __name__ == "__main__":
    main()
