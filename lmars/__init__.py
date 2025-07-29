"""
L-MARS - Legal Multi-Agent Workflow for Orchestrated Reasoning and Agentic Search

A modern legal research system using LangGraph with structured output.
Features Query, Search, Judge, and Summary agents working together.
"""

from .graph import create_legal_mind_graph, LegalMindGraph
from .agents import (
    QueryAgent, SearchAgent, JudgeAgent, SummaryAgent,
    FollowUpQuestion, QueryGeneration, SearchResult, 
    JudgmentResult, FinalAnswer
)

__version__ = "2.0.0"

__all__ = [
    # Main system
    "create_legal_mind_graph",
    "LegalMindGraph", 
    
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
    "FinalAnswer"
]