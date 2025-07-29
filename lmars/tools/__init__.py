"""
Tools package for legal research system.
Contains search tools used by the multi-agent system.
"""
from .courtlistener_tool import find_legal_cases, search_legal_cases
from .serper_search_tool import search_serper_web, search_serper_with_content

__all__ = [
    "find_legal_cases", 
    "search_legal_cases", 
    "search_serper_web", 
    "search_serper_with_content"
]