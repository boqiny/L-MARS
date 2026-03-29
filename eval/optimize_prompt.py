#!/usr/bin/env python3
"""Optimize the L-MARS bar-exam synthesis prompt with GEPA.

Strategy
--------
GEPA iterates over candidate *system prompts*.  For each candidate it runs
batch inference (litellm, gpt-4o-mini) on the training set and reflects on
failures to propose a better prompt.

To keep cost low we **pre-retrieve** all Serper snippets once (cached to
eval/cache/serper/) before the GEPA loop starts.  GEPA then only pays for
N_train × max_evals GPT calls — no extra Serper credits.

The training set is drawn from questions *not* in the n=50 eval sample so
the evaluation on that set remains uncontaminated.

Usage
-----
    python3 -m eval.optimize_prompt [--n-train 30] [--max-evals 30] [--out PATH]

After the run the best system prompt is written to the path given by --out
(default: prompts/barexam_system_prompt.txt).  To use it in eval:

    # in eval/lmars_eval.py, replace create_workflow(...) with:
    _SP = Path(_ROOT / "prompts" / "barexam_system_prompt.txt").read_text()
    _WORKFLOW = create_workflow(..., system_prompt=_SP)
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval.api_utils import parse_answer_letter, serper_search
from eval.config import BAREXAM_CSV, LEGALSEARCHQA_JSON, LMARS_MODEL_ID, PILOT_SEED
from eval.data_loader import load_barexam_labeled, load_legalsearchqa

# ── seed system prompt ────────────────────────────────────────────────────────
SEED_SYSTEM_PROMPT = """\
You are a bar exam expert. A user will provide retrieved legal sources and a
multiple-choice question. Use the retrieved sources together with your legal
knowledge to select the single best answer.

Rules:
1. Your response MUST start with exactly one letter on the first line: A, B, C, or D.
2. Follow with a concise 1-2 sentence rationale citing the controlling legal rule.
3. Bar exam questions test specific exceptions and nuances — do not just apply
   the most common general rule if the specific facts call for an exception.

Response format:
[Letter]
[Brief rationale]"""

OPTIMIZED_PROMPT_DEFAULT = str(_ROOT / "prompts" / "barexam_system_prompt.txt")


# ── custom evaluator ──────────────────────────────────────────────────────────
class BarExamEvaluator:
    """Use parse_answer_letter for robust A/B/C/D extraction."""

    def __call__(self, data: dict, response: str):
        from gepa.adapters.default_adapter.default_adapter import EvaluationResult

        pred = parse_answer_letter(response)
        gold = data["answer"]
        is_correct = pred == gold

        if is_correct:
            feedback = f"Correct — model selected {gold}."
        else:
            ctx = data.get("additional_context", {})
            feedback = (
                f"WRONG — model selected '{pred}', correct answer is '{gold}'. "
                f"Subject: {ctx.get('subject', '?')}. "
                f"Question: {ctx.get('question_preview', '')}…"
            )

        return EvaluationResult(score=1.0 if is_correct else 0.0, feedback=feedback)


def build_user_message(row: dict, evidence: str) -> str:
    """Build the fixed-structure user message (evidence + full MCQ)."""
    mcq = (
        f"{row['prompt']}\n\n"
        f"{row['question']}\n\n"
        f"A. {row['choice_a']}\n"
        f"B. {row['choice_b']}\n"
        f"C. {row['choice_c']}\n"
        f"D. {row['choice_d']}"
    )
    return f"Retrieved Evidence:\n{evidence}\n\nQuestion:\n{mcq}"


def main() -> None:
    parser = argparse.ArgumentParser(description="GEPA prompt optimiser for L-MARS bar exam")
    parser.add_argument(
        "--dataset",
        choices=["barexam", "legalsearchqa"],
        default="barexam",
        help="Which dataset to use for optimisation (default: barexam)",
    )
    parser.add_argument(
        "--n-train", type=int, default=30,
        help="Number of training questions (not from n50 eval set). Default: 30",
    )
    parser.add_argument(
        "--n-val", type=int, default=10,
        help="Number of validation questions. Default: 10",
    )
    parser.add_argument(
        "--max-evals", type=int, default=30,
        help="Max GEPA metric calls (prompt candidates evaluated). Default: 30",
    )
    parser.add_argument(
        "--out", type=str, default=OPTIMIZED_PROMPT_DEFAULT,
        help="Where to save the optimised system prompt",
    )
    parser.add_argument(
        "--reflection-lm", type=str, default="openai/gpt-4o",
        help="LLM used by GEPA for reflection / prompt mutation. Default: openai/gpt-4o",
    )
    args = parser.parse_args()

    # ── load & split data ─────────────────────────────────────────────────────
    if args.dataset == "legalsearchqa":
        all_rows = load_legalsearchqa(LEGALSEARCHQA_JSON)
        print(f"Loaded {len(all_rows)} LegalSearchQA questions.")
    else:
        all_rows = load_barexam_labeled(BAREXAM_CSV)
        print(f"Loaded {len(all_rows)} bar exam questions.")

    # Reproduce the n50 eval sample to exclude those IDs (or fewer for small datasets)
    random.seed(PILOT_SEED)
    n_eval_holdout = min(50, len(all_rows) // 3)  # hold out at most 1/3 for eval
    eval_ids = {str(r["example_id"] or r["idx"]) for r in random.sample(all_rows, n_eval_holdout)}

    pool = [r for r in all_rows if str(r["example_id"] or r["idx"]) not in eval_ids]
    print(f"Pool after removing n{n_eval_holdout} eval set: {len(pool)} questions.")

    n_needed = args.n_train + args.n_val
    if n_needed > len(pool):
        parser.error(f"--n-train + --n-val ({n_needed}) exceeds pool size ({len(pool)})")

    random.seed(99)  # different seed to avoid any overlap pattern
    sample = random.sample(pool, n_needed)
    train_rows = sample[: args.n_train]
    val_rows   = sample[args.n_train :]
    print(f"Train: {len(train_rows)}  Val: {len(val_rows)}\n")

    # ── pre-retrieve Serper evidence (cached automatically) ───────────────────
    print("Pre-retrieving Serper evidence (uses local cache)…")
    train_evidence: dict[str, str] = {}
    val_evidence:   dict[str, str] = {}

    for label, rows, store in [("train", train_rows, train_evidence), ("val", val_rows, val_evidence)]:
        for i, row in enumerate(rows, 1):
            row_id = str(row["example_id"] or row["idx"])
            q = f"US law bar exam: {row['question'][:200]}"
            ev = serper_search(q, k=5)
            store[row_id] = ev
            print(f"  [{label}] {i}/{len(rows)} id={row_id}", end="\r")
        print()

    # ── build GEPA datasets ───────────────────────────────────────────────────
    def to_inst(row: dict, evidence_store: dict) -> dict:
        row_id = str(row["example_id"] or row["idx"])
        return {
            "input": build_user_message(row, evidence_store[row_id]),
            "additional_context": {
                "subject": row["subject"],
                "question_preview": row["question"][:200],
            },
            "answer": row["answer"].strip().upper(),
        }

    trainset = [to_inst(r, train_evidence) for r in train_rows]
    valset   = [to_inst(r, val_evidence)   for r in val_rows]

    # ── run GEPA ──────────────────────────────────────────────────────────────
    from gepa import optimize

    # litellm model name format (not langchain "openai:…")
    task_lm = LMARS_MODEL_ID.replace("openai:", "openai/")

    print(f"\nStarting GEPA optimisation…")
    print(f"  task_lm      : {task_lm}")
    print(f"  reflection_lm: {args.reflection_lm}")
    print(f"  max_evals    : {args.max_evals}")
    print(f"  train/val    : {len(trainset)}/{len(valset)}\n")

    result = optimize(
        seed_candidate={"system_prompt": SEED_SYSTEM_PROMPT},
        trainset=trainset,
        valset=valset,
        task_lm=task_lm,
        evaluator=BarExamEvaluator(),
        reflection_lm=args.reflection_lm,
        max_metric_calls=args.max_evals,
        display_progress_bar=True,
    )

    # ── report & save ─────────────────────────────────────────────────────────
    best_prompt = result.best_candidate["system_prompt"]
    best_score  = result.val_aggregate_scores[result.best_idx]
    print(f"\n{'='*60}")
    print(f"GEPA done.  Best val score: {best_score:.1%}  (candidate #{result.best_idx})")
    print(f"All val scores: {[round(s, 2) for s in result.val_aggregate_scores]}")
    print(f"{'='*60}")
    print("\nOptimised system prompt:")
    print("-" * 60)
    print(best_prompt)
    print("-" * 60)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(best_prompt, encoding="utf-8")
    print(f"\nSaved → {out_path}")
    print(
        "\nTo use in eval/lmars_eval.py, replace the _WORKFLOW definition with:\n"
        "    _SP = Path(_ROOT / 'prompts' / 'barexam_system_prompt.txt').read_text()\n"
        "    _WORKFLOW = create_workflow(..., system_prompt=_SP)"
    )


if __name__ == "__main__":
    main()
