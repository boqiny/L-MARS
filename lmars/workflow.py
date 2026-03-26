from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal

from .agents import QueryAgent, SearchAgent, SummaryAgent, to_serializable_search_results


@dataclass
class WorkflowConfig:
    mode: Literal["simple"] = "simple"
    llm_model: str = "openai:gpt-4o-mini"
    use_deep_content: bool = False
    max_results: int = 5
    judge_model: str | None = None
    max_iterations: int = 1
    enable_tracking: bool = False
    enable_offline_rag: bool = False
    enable_courtlistener: bool = False


class LMarsWorkflow:
    """Minimal single-turn workflow: QueryAgent -> SearchAgent -> SummaryAgent."""

    def __init__(self, config: WorkflowConfig | None = None):
        self.config = config or WorkflowConfig()
        self.query_agent = QueryAgent()
        self.search_agent = SearchAgent(
            use_deep_content=self.config.use_deep_content,
            max_results=self.config.max_results,
        )
        self.summary_agent = SummaryAgent(model_id=self.config.llm_model)

    def run(
        self,
        query: str,
        example_id: str,
        use_cache: bool,
        cache_dir: str,
        seed: int,
    ) -> Dict[str, Any]:
        structured_query = self.query_agent.build_query(query)
        retrieved = self.search_agent.search(
            example_id=example_id,
            query=structured_query.query,
            use_cache=use_cache,
            cache_dir=cache_dir,
        )
        summary = self.summary_agent.generate_final_answer(query, retrieved, seed=seed)

        return {
            "query": query,
            "structured_query": structured_query.query,
            "retrieved": to_serializable_search_results(retrieved),
            "final_answer": summary["final_answer"].answer,
            "rationale": summary["final_answer"].rationale,
            "prompt_used": summary["prompt"],
            "mode": "simple",
        }


def create_workflow(
    mode: Literal["simple"] = "simple",
    llm_model: str = "openai:gpt-4o-mini",
    use_deep_content: bool = False,
    max_results: int = 5,
    **_: Any,
) -> LMarsWorkflow:
    if mode != "simple":
        raise ValueError("Only single-turn simple mode is supported.")

    return LMarsWorkflow(
        WorkflowConfig(
            mode="simple",
            llm_model=llm_model,
            use_deep_content=use_deep_content,
            max_results=max_results,
        )
    )
