from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import yaml

from data.lexam.loader import load_lexam
from lmars.workflow import create_workflow


def _resolve_dataset_path(dataset: str, dataset_path: str | None) -> str:
    name = dataset.lower()
    if name not in {"leexam", "lexam"}:
        raise ValueError("Only LEXam is supported. Use --dataset leexam.")
    return dataset_path or "data/lexam/lexam.jsonl"


def _load_retriever_config(path: str) -> Dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Retriever config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_single_turn_pipeline(
    dataset: str,
    model: str,
    retriever_config: str,
    use_cache: bool,
    output: str,
    seed: int,
    cache_dir: str,
    dataset_path: str | None = None,
) -> List[Dict]:
    path = _resolve_dataset_path(dataset, dataset_path)
    examples = load_lexam(path)
    retriever_cfg = _load_retriever_config(retriever_config)

    workflow = create_workflow(
        mode="simple",
        llm_model=model,
        use_deep_content=bool(retriever_cfg.get("use_deep_content", False)),
        max_results=int(retriever_cfg.get("max_results", 5)),
    )

    rows: List[Dict] = []
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            result = workflow.run(
                query=ex["input"],
                example_id=ex["id"],
                use_cache=use_cache,
                cache_dir=cache_dir,
                seed=seed,
            )
            row = {
                "id": ex["id"],
                "question": ex["input"],
                "pred": result["final_answer"],
                "gold": ex.get("gold"),
                "log": {
                    "retrieved": result["retrieved"],
                    "prompt": result["prompt_used"],
                    "mode": "simple",
                    "seed": seed,
                    "model_id": model,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)

    return rows
