"""
L-MARS - Legal Multi-Agent Workflow for Orchestrated Reasoning and Agentic Search

A modern legal research system with two modes:
- Simple Mode: Single-turn retrieval and response (fast)
- Multi-Turn Mode: Iterative refinement with agents (thorough)
"""

from .workflow import LMarsWorkflow, create_workflow, WorkflowConfig
from .agents import (
    QueryAgent, SearchAgent, JudgeAgent, SummaryAgent,
    FollowUpQuestion, QueryGeneration, SearchResult, 
    JudgmentResult, FinalAnswer
)

__version__ = "2.0.0"

__all__ = [
    # Main workflow system
    "LMarsWorkflow",
    "create_workflow",
    "WorkflowConfig",
    
    # Agents
    "QueryAgent",
    "SearchAgent",
    "JudgeAgent", 
    "SummaryAgent",
    
    # Data models
    "FollowUpQuestion",
    "QueryGeneration",
    "SearchResult",
    "JudgmentResult", 
    "FinalAnswer"
]