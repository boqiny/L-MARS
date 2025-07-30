"""
L-MARS - Legal Multi-Agent Workflow for Orchestrated Reasoning and Agentic Search

A modern legal research system using LangGraph with structured output.
Features Query, Search, Judge, and Summary agents working together.
"""

from .graph import create_legal_mind_graph, LMarsGraph
from .agents import (
    QueryAgent, SearchAgent, JudgeAgent, SummaryAgent,
    FollowUpQuestion, QueryGeneration, SearchResult, 
    JudgmentResult, FinalAnswer, Person, QueryResult,
    FollowUpQuestionList, QueryGenerationList
)

__version__ = "1.0.0"

__all__ = [
    # Main system
    "create_legal_mind_graph",
    "LMarsGraph", 
    
    # Agents
    "QueryAgent",
    "SearchAgent",
    "JudgeAgent", 
    "SummaryAgent",
    
    # Structured output models
    "FollowUpQuestion",
    "QueryGeneration",
    "SearchResult",
    "JudgmentResult", 
    "FinalAnswer",
    "Person",
    "QueryResult",
    "FollowUpQuestionList",
    "QueryGenerationList"
]