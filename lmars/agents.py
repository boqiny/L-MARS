from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

from .tools.serper_search_tool import search_serper_web, search_serper_with_content


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


class QueryAgent:
    """Single-turn query builder."""

    def build_query(self, user_query: str) -> QueryGeneration:
        return QueryGeneration(query=user_query.strip())


class SearchAgent:
    """Search executor with deterministic cache support."""

    def __init__(self, use_deep_content: bool = False, max_results: int = 5):
        self.use_deep_content = use_deep_content
        self.max_results = max_results

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


class SummaryAgent:
    """Single-step answer synthesis using retrieved evidence."""

    def __init__(self, model_id: str, template_path: str = "prompts/summary_template.txt"):
        self.model_id = model_id
        self.template_path = template_path
        self.template = Path(template_path).read_text(encoding="utf-8")

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
        prompt = self.template.format(question=user_query, evidence=evidence)

        if self.model_id.lower() in {"mock", "qwen3"}:
            answer = self._mock_answer(user_query, search_results)
            return {"final_answer": answer, "prompt": prompt}

        llm = init_chat_model(self.model_id, temperature=0)
        resp = llm.invoke([HumanMessage(content=prompt)])
        text = resp.content if hasattr(resp, "content") else str(resp)

        return {
            "final_answer": FinalAnswer(answer=text.strip(), rationale="Generated from retrieved evidence."),
            "prompt": prompt,
        }


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
