from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from .tools.serper_search_tool import search_serper_web, search_serper_with_content
from .tools.tavily_search_tool import search_tavily


@dataclass
class QueryGeneration:
    query: str
    query_type: str = "web_search"
    priority: str = "high"


@dataclass
class SearchResult:
    source: str
    title: str
    content: str
    url: str | None = None


@dataclass
class FinalAnswer:
    answer: str
    rationale: str


@dataclass
class JudgeDecision:
    sufficient: bool
    reason: str
    next_query: str | None = None


# Patterns that indicate a result has no usable content
_BLOCKED_PATTERNS = [
    "HTTP error occurred",
    "403 Client Error",
    "404 Client Error",
    "Error: Connection error",
    "Request timed out",
    "JavaScript is disabled",
    "Could not extract content",
    "No search results found",
    "Error during Serper search",
    "SEARCH_ERROR",
    "API request failed",
]

_JUDGE_SYSTEM = """\
You are a legal research quality judge for US bar exam questions.

You will receive a bar exam multiple-choice question and a list of retrieved \
search result snippets. Your job is to decide whether the retrieved evidence \
is sufficient to reliably answer the question.

"Sufficient" means: at least one result contains specific legal rules, statutes, \
case holdings, or doctrinal exceptions that directly address the core legal issue \
tested by the question — not just a restatement of the facts.

If the evidence is NOT sufficient, provide a better search query that targets \
primary legal sources (e.g., specific rule names, statute citations, or doctrine \
names) rather than generic question text.

Respond with valid JSON only, no markdown fences:
{"sufficient": true_or_false, "reason": "one sentence", "next_query": "refined query (omit if sufficient)"}"""


def is_blocked(result: SearchResult) -> bool:
    """Return True if the result has no usable content (blocked, errored, or empty)."""
    if not result.content or len(result.content.strip()) < 30:
        return True
    return any(pat in result.content for pat in _BLOCKED_PATTERNS)


class QueryAgent:
    """Single-turn query builder."""

    def build_query(self, user_query: str) -> QueryGeneration:
        return QueryGeneration(query=user_query.strip())


class SearchAgent:
    """Search executor with deterministic cache support.

    search_backend options:
    - "serper"        : Serper API, snippet-only (fast, cached locally)
    - "serper_deep"   : Serper API + live webpage scraping
    - "tavily"        : Tavily API, AI-optimised extracted content (recommended)
    """

    def __init__(
        self,
        use_deep_content: bool = False,
        max_results: int = 5,
        search_backend: str = "serper",
    ):
        self.use_deep_content = use_deep_content
        self.max_results = max_results
        self.search_backend = search_backend

    def search(
        self,
        example_id: str,
        query: str,
        use_cache: bool,
        cache_dir: str,
    ) -> List[SearchResult]:
        if use_cache:
            cache_file = Path(cache_dir) / f"{example_id}.json"
            if not cache_file.exists():
                raise FileNotFoundError(
                    f"Cache miss for id={example_id}. Expected {cache_file}. "
                    "Generate cache with eval/cache_generator.py or disable --use-cache."
                )
            with cache_file.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            return [SearchResult(**item) for item in payload.get("retrieved", [])]

        if self.search_backend == "tavily":
            raw = search_tavily(query, max_results=self.max_results)
            return self._parse_tavily_output(raw, query)

        raw = (
            search_serper_with_content(query, max_results=self.max_results)
            if self.use_deep_content
            else search_serper_web(query, max_results=self.max_results)
        )
        return self._parse_serper_output(raw, query)

    def _parse_serper_output(self, raw_results: str, query: str) -> List[SearchResult]:
        lines = raw_results.split("\n")
        results: List[SearchResult] = []

        current_title = ""
        current_url = ""
        current_site = ""
        current_summary = ""
        current_content = ""

        for line in lines:
            line = line.strip()
            if line and line[0].isdigit() and ". " in line:
                if current_title and (current_summary or current_content):
                    results.append(
                        SearchResult(
                            source=f"Web Search - {current_site}" if current_site else "Web Search",
                            title=current_title,
                            content=current_content or current_summary,
                            url=current_url or None,
                        )
                    )
                current_title = line.split(". ", 1)[1]
                current_url = ""
                current_site = ""
                current_summary = ""
                current_content = ""
            elif line.startswith("URL: "):
                current_url = line[5:]
            elif line.startswith("Site: "):
                current_site = line[6:]
            elif line.startswith("Summary: "):
                current_summary = line[9:]
            elif line.startswith("Content: "):
                current_content = line[9:]

        if current_title and (current_summary or current_content):
            results.append(
                SearchResult(
                    source=f"Web Search - {current_site}" if current_site else "Web Search",
                    title=current_title,
                    content=current_content or current_summary,
                    url=current_url or None,
                )
            )

        if not results:
            results.append(
                SearchResult(
                    source="Web Search",
                    title=f"Search results for: {query}",
                    content=raw_results[:1200],
                    url=None,
                )
            )

        return results

    def _parse_tavily_output(self, raw_results: str, query: str) -> List[SearchResult]:
        """Parse the formatted string from search_tavily() into SearchResult objects."""
        results: List[SearchResult] = []

        if raw_results.startswith("Error") or raw_results.startswith("No Tavily"):
            results.append(
                SearchResult(
                    source="Tavily Search",
                    title=f"Search results for: {query}",
                    content=raw_results,
                    url=None,
                )
            )
            return results

        current_title = ""
        current_url = ""
        current_content = ""

        for line in raw_results.split("\n"):
            line = line.strip()
            if line and line[0].isdigit() and ". " in line:
                if current_title and current_content:
                    results.append(
                        SearchResult(
                            source="Tavily Search",
                            title=current_title,
                            content=current_content,
                            url=current_url or None,
                        )
                    )
                current_title = line.split(". ", 1)[1]
                current_url = ""
                current_content = ""
            elif line.startswith("URL: "):
                current_url = line[5:]
            elif line.startswith("Content: "):
                current_content = line[9:]

        if current_title and current_content:
            results.append(
                SearchResult(
                    source="Tavily Search",
                    title=current_title,
                    content=current_content,
                    url=current_url or None,
                )
            )

        if not results:
            results.append(
                SearchResult(
                    source="Tavily Search",
                    title=f"Search results for: {query}",
                    content=raw_results[:1200],
                    url=None,
                )
            )

        return results


class SummaryAgent:
    """Single-step answer synthesis using retrieved evidence.

    Two operating modes:
    - Template mode (default): fills ``{question}`` and ``{evidence}`` placeholders
      in a template file and sends a single HumanMessage.
    - System-prompt mode: when ``system_prompt`` is provided, sends a SystemMessage
      with the instructions and a HumanMessage with the evidence + question.  This
      is the mode used after GEPA prompt optimisation.
    """

    def __init__(
        self,
        model_id: str,
        template_path: str = "prompts/summary_template.txt",
        system_prompt: str | None = None,
    ):
        self.model_id = model_id
        self.system_prompt = system_prompt
        if system_prompt is None:
            self.template_path = template_path
            self.template = Path(template_path).read_text(encoding="utf-8")
        else:
            self.template_path = None
            self.template = None

    def _mock_answer(self, user_query: str, search_results: List[SearchResult]) -> FinalAnswer:
        evidence_blob = "\n".join(r.content for r in search_results)
        match = re.search(r"\bANSWER\s*:\s*([A-Za-z0-9]+)\b", evidence_blob, re.IGNORECASE)
        answer = match.group(1).upper() if match else "UNKNOWN"
        rationale = "Extracted deterministic answer token from cached evidence."
        return FinalAnswer(answer=answer, rationale=rationale)

    def generate_final_answer(self, user_query: str, search_results: List[SearchResult], seed: int) -> Dict[str, Any]:
        evidence = "\n\n".join(
            [f"Source: {r.source}\nTitle: {r.title}\nContent: {r.content}" for r in search_results]
        )

        if self.system_prompt is not None:
            # System-prompt mode: instructions in system, evidence+question in user
            user_content = f"Retrieved Evidence:\n{evidence}\n\nQuestion:\n{user_query}"
            messages = [SystemMessage(content=self.system_prompt), HumanMessage(content=user_content)]
            prompt_logged = user_content
        else:
            # Template mode
            prompt_logged = self.template.format(question=user_query, evidence=evidence)
            messages = [HumanMessage(content=prompt_logged)]

        if self.model_id.lower() in {"mock", "qwen3"}:
            answer = self._mock_answer(user_query, search_results)
            return {"final_answer": answer, "prompt": prompt_logged}

        llm = init_chat_model(self.model_id, temperature=0)
        resp = llm.invoke(messages)
        text = resp.content if hasattr(resp, "content") else str(resp)

        return {
            "final_answer": FinalAnswer(answer=text.strip(), rationale="Generated from retrieved evidence."),
            "prompt": prompt_logged,
        }


class JudgeAgent:
    """Evaluates retrieval quality and proposes a refined query when evidence is weak.

    Strategy
    --------
    1. Pre-filter: deterministically remove blocked/errored results (no LLM cost).
    2. LLM judge: ask GPT-4o-mini whether the remaining snippets contain specific
       legal doctrine sufficient to answer the question.
    3. If not sufficient, the judge proposes a better search query targeting
       primary legal sources rather than repackaged question text.
    """

    def __init__(self, model_id: str = "openai:gpt-4o-mini"):
        self.model_id = model_id

    def evaluate(
        self,
        question: str,
        results: List[SearchResult],
    ) -> JudgeDecision:
        valid = [r for r in results if not is_blocked(r)]

        if not valid:
            return JudgeDecision(
                sufficient=False,
                reason="All retrieved results were blocked or inaccessible.",
            )

        evidence_summary = "\n\n".join(
            f"[{i + 1}] {r.title}\n{r.content[:400]}"
            for i, r in enumerate(valid)
        )
        user_content = f"Question:\n{question}\n\nRetrieved results ({len(valid)} non-blocked):\n{evidence_summary}"

        llm = init_chat_model(self.model_id, temperature=0)
        resp = llm.invoke(
            [SystemMessage(content=_JUDGE_SYSTEM), HumanMessage(content=user_content)]
        )
        text = resp.content if hasattr(resp, "content") else str(resp)

        try:
            # Strip markdown fences if model adds them anyway
            clean = re.sub(r"```(?:json)?|```", "", text).strip()
            data = json.loads(clean)
            return JudgeDecision(
                sufficient=bool(data.get("sufficient", False)),
                reason=data.get("reason", ""),
                next_query=data.get("next_query") or None,
            )
        except (json.JSONDecodeError, KeyError):
            return JudgeDecision(
                sufficient=False,
                reason=f"Judge parse error — treating as insufficient. Raw: {text[:120]}",
            )


def to_serializable_search_results(results: List[SearchResult]) -> List[Dict[str, Any]]:
    return [
        {
            "source": r.source,
            "title": r.title,
            "content": r.content,
            "url": r.url,
        }
        for r in results
    ]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
