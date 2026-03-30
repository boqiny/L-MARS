from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Literal

from .agents import (
    JudgeAgent,
    QueryAgent,
    SearchAgent,
    SummaryAgent,
    is_blocked,
    to_serializable_search_results,
)


@dataclass
class WorkflowConfig:
    mode: Literal["simple"] = "simple"
    llm_model: str = "openai:gpt-4o-mini"
    judge_model: str = "openai:gpt-4o-mini"
    # "serper" | "serper_deep" | "tavily"
    search_backend: str = "serper"
    use_deep_content: bool = False
    max_results: int = 5
    # Multi-turn: 1 = single-pass (original behaviour); >1 enables JudgeAgent loop.
    max_turns: int = 1
    enable_offline_rag: bool = False
    enable_courtlistener: bool = False
    template_path: str = "prompts/summary_template.txt"
    # When set, overrides template_path — system prompt mode (GEPA-optimised).
    system_prompt: str | None = None


class LMarsWorkflow:
    """QueryAgent → [JudgeAgent loop] → SummaryAgent.

    When ``max_turns == 1`` the workflow is identical to the original single-pass
    behaviour.  When ``max_turns > 1`` a JudgeAgent evaluates each round of
    retrieved evidence and can trigger a refined search query, repeating up to
    ``max_turns`` times before the SummaryAgent synthesises the final answer.

    Multi-turn loop
    ---------------
    1. Search with current query.
    2. Pre-filter: drop results with blocked / error content (no LLM cost).
    3. JudgeAgent: is the accumulated valid evidence sufficient?
       - Yes → stop early, proceed to synthesis.
       - No  → use judge's ``next_query`` (or a fallback enrichment) for the
               next turn.
    4. Synthesise over all valid results (fallback to all results if none valid).
    """

    def __init__(self, config: WorkflowConfig | None = None):
        self.config = config or WorkflowConfig()
        self.query_agent = QueryAgent()
        self.search_agent = SearchAgent(
            use_deep_content=self.config.use_deep_content,
            max_results=self.config.max_results,
            search_backend=self.config.search_backend,
        )
        self.summary_agent = SummaryAgent(
            model_id=self.config.llm_model,
            template_path=self.config.template_path,
            system_prompt=self.config.system_prompt,
        )
        if self.config.max_turns > 1:
            self.judge_agent: JudgeAgent | None = JudgeAgent(
                model_id=self.config.judge_model
            )
        else:
            self.judge_agent = None

    def run(
        self,
        query: str,
        example_id: str,
        use_cache: bool,
        cache_dir: str,
        seed: int,
        search_query: str | None = None,
    ) -> Dict[str, Any]:
        """Run the workflow and return a result dict.

        Args:
            query:        Full synthesis prompt (MCQ with answer choices).
            example_id:   Unique row ID used as cache key.
            use_cache:    Load results from a pre-built cache instead of calling Serper.
            cache_dir:    Directory for the per-question cache files.
            seed:         RNG seed forwarded to the SummaryAgent.
            search_query: A shorter, focused Serper query (no answer choices).
                          Defaults to ``query`` when omitted.
        """
        initial_query = search_query if search_query is not None else query
        current_query = initial_query
        all_results = []
        seen_keys: set[str] = set()
        search_history: list[str] = []
        judge_log: list[dict] = []

        for turn in range(self.config.max_turns):
            search_history.append(current_query)
            structured = self.query_agent.build_query(current_query)

            # Use a per-turn cache key so different turns (different queries) don't
            # collide.  Turn 0 uses the bare example_id for backward compatibility.
            turn_id = example_id if turn == 0 else f"{example_id}_t{turn}"
            retrieved = self.search_agent.search(
                example_id=turn_id,
                query=structured.query,
                use_cache=use_cache,
                cache_dir=cache_dir,
            )

            # Deduplicate across turns by URL (or title as fallback)
            for r in retrieved:
                key = r.url or r.title
                if key not in seen_keys:
                    seen_keys.add(key)
                    all_results.append(r)

            # No judge on the final turn — just synthesise whatever we have
            if self.judge_agent is None or turn == self.config.max_turns - 1:
                break

            valid_so_far = [r for r in all_results if not is_blocked(r)]
            decision = self.judge_agent.evaluate(query, valid_so_far)

            judge_log.append(
                {
                    "turn": turn,
                    "query": current_query,
                    "valid_results": len(valid_so_far),
                    "sufficient": decision.sufficient,
                    "reason": decision.reason,
                    "next_query": decision.next_query,
                }
            )
            print(
                f"  [judge t{turn}] sufficient={decision.sufficient}  "
                f"valid={len(valid_so_far)}  reason={decision.reason!r}",
                file=sys.stderr,
            )

            if decision.sufficient:
                break

            # Build next query: prefer judge's suggestion, fall back to enrichment
            if decision.next_query:
                current_query = decision.next_query
            else:
                # Generic enrichment: append legal vocabulary to diversify results
                current_query = initial_query + " legal rule statute doctrine"

        # Use only non-blocked results for synthesis; fall back to all if none pass
        valid_results = [r for r in all_results if not is_blocked(r)]
        synthesis_results = valid_results if valid_results else all_results

        summary = self.summary_agent.generate_final_answer(query, synthesis_results, seed=seed)

        return {
            "query": query,
            "search_history": search_history,
            "retrieved": to_serializable_search_results(all_results),
            "valid_retrieved": to_serializable_search_results(synthesis_results),
            "final_answer": summary["final_answer"].answer,
            "rationale": summary["final_answer"].rationale,
            "prompt_used": summary["prompt"],
            "judge_log": judge_log,
            "turns_used": len(search_history),
            "mode": "multi_turn" if self.config.max_turns > 1 else "simple",
        }


def create_workflow(
    mode: Literal["simple"] = "simple",
    llm_model: str = "openai:gpt-4o-mini",
    judge_model: str = "openai:gpt-4o-mini",
    search_backend: str = "serper",
    use_deep_content: bool = False,
    max_results: int = 5,
    max_turns: int = 1,
    template_path: str = "prompts/summary_template.txt",
    system_prompt: str | None = None,
    **_: Any,
) -> LMarsWorkflow:
    return LMarsWorkflow(
        WorkflowConfig(
            mode="simple",
            llm_model=llm_model,
            judge_model=judge_model,
            search_backend=search_backend,
            use_deep_content=use_deep_content,
            max_results=max_results,
            max_turns=max_turns,
            template_path=template_path,
            system_prompt=system_prompt,
        )
    )
