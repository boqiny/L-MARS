from __future__ import annotations

from typing import Dict, List

from lmars.pipeline_runner import run_single_turn_pipeline


def run_qwen3(
    dataset: str,
    output: str,
    use_cache: bool,
    retriever_config: str,
    seed: int,
    cache_dir: str,
    dataset_path: str | None = None,
) -> List[Dict]:
    return run_single_turn_pipeline(
        dataset=dataset,
        dataset_path=dataset_path,
        model="qwen3",
        retriever_config=retriever_config,
        use_cache=use_cache,
        output=output,
        seed=seed,
        cache_dir=cache_dir,
    )
