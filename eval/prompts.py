"""All prompt templates for the L-MARS baseline evaluation pipeline."""
from __future__ import annotations

# ── Bar Exam QA ───────────────────────────────────────────────────────────────

BAREXAM_ZERO_SHOT = """\
You are a legal expert. Answer the following bar exam question by selecting the correct option.

Context: {prompt}

Question: {question}

Options:
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Respond with ONLY the letter of the correct answer (A, B, C, or D). Do not explain.\
"""

BAREXAM_COT = """\
You are a legal expert. Answer the following bar exam question.

Context: {prompt}

Question: {question}

Options:
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Think step by step:
1. Identify the legal issue being tested.
2. Recall the relevant legal rule or principle.
3. Apply the rule to the facts.
4. Eliminate incorrect options.

After your reasoning, state your final answer on a new line in the exact format:
ANSWER: X

where X is A, B, C, or D.\
"""

BAREXAM_RAG = """\
You are a legal expert. Answer the following bar exam question using the provided reference materials.

Reference Materials:
{retrieved_snippets}

Context: {prompt}

Question: {question}

Options:
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

Based on the reference materials and your legal knowledge, select the correct answer.
Respond with ONLY the letter of the correct answer (A, B, C, or D). Do not explain.\
"""
