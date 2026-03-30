"""L-MARS minimal single-turn legal QA pipeline."""

from .workflow import LMarsWorkflow, WorkflowConfig, create_workflow
from .agents import FinalAnswer, QueryAgent, QueryGeneration, SearchAgent, SearchResult, SummaryAgent

__version__ = "3.0.0"

__all__ = [
    "LMarsWorkflow",
    "WorkflowConfig",
    "create_workflow",
    "QueryAgent",
    "QueryGeneration",
    "SearchAgent",
    "SearchResult",
    "SummaryAgent",
    "FinalAnswer",
]
