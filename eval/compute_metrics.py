from __future__ import annotations

import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage


def _normalize_label(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().upper())


def compute_f1(preds: List[str], golds: List[str]) -> float:
    """Micro-F1 for single-label short-answer/MCQ style outputs."""
    if len(preds) != len(golds):
        raise ValueError("preds and golds must have the same length")
    if not preds:
        return 0.0

    tp = 0
    fp = 0
    fn = 0

    for pred, gold in zip(preds, golds):
        p = _normalize_label(pred)
        g = _normalize_label(gold)
        if p == g:
            tp += 1
        else:
            fp += 1
            fn += 1

    denom = (2 * tp) + fp + fn
    return 0.0 if denom == 0 else (2 * tp) / denom


def _parse_judge_response(text: str) -> Dict[str, str]:
    rating_match = re.search(r"RATING\s*:\s*(.+)", text, flags=re.IGNORECASE)
    reason_match = re.search(r"REASON\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    return {
        "rating": rating_match.group(1).strip() if rating_match else "unknown",
        "reason": reason_match.group(1).strip() if reason_match else text.strip(),
    }


def run_llm_as_judge(pred_jsonl: str, sample_n: int, llm_model: str) -> List[Dict]:
    prompt_template = Path("prompts/llm_judge.txt").read_text(encoding="utf-8")

    rows = []
    with Path(pred_jsonl).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    random.seed(7)
    if sample_n < len(rows):
        rows = random.sample(rows, sample_n)

    judged: List[Dict] = []
    llm = None if llm_model.lower() == "mock" else init_chat_model(llm_model, temperature=0)

    for row in rows:
        question = row.get("question", "")
        pred = row.get("pred", "")
        gold = row.get("gold", "")
        prompt = prompt_template.format(question=question, pred=pred, gold=gold)

        if llm is None:
            rating = "good" if _normalize_label(pred) == _normalize_label(gold) else "poor"
            parsed = {"rating": rating, "reason": "Deterministic mock judge based on exact match."}
        else:
            response = llm.invoke([HumanMessage(content=prompt)])
            content = response.content if hasattr(response, "content") else str(response)
            parsed = _parse_judge_response(content)

        judged.append(
            {
                "id": row.get("id"),
                "rating": parsed["rating"],
                "reason": parsed["reason"],
                "model_id": llm_model,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    return judged
