"""
Result Logger for L-MARS
Logs all backend processes, search results, and LLM interactions to files
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid


class ResultLogger:
    """Comprehensive logger for all L-MARS operations."""
    
    def __init__(self, results_dir: str = "results", session_id: str = None):
        """
        Initialize the result logger.
        
        Args:
            results_dir: Directory to save result logs
            session_id: Unique session identifier
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate session ID
        self.session_id = session_id or str(uuid.uuid4())[:8]
        
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.results_dir / f"lmars_log_{timestamp}_{self.session_id}.json"
        
        # Initialize log structure
        self.log_data = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "query": "",
            "configuration": {},
            "process_log": [],
            "search_queries": [],
            "search_results": [],
            "llm_interactions": [],
            "final_answer": None,
            "metadata": {
                "total_search_results": 0,
                "sources_used": [],
                "duration_seconds": 0,
                "errors": []
            }
        }
        
        self.start_time = datetime.now()
        
    def set_query(self, query: str):
        """Set the main query being processed."""
        self.log_data["query"] = query
        self._add_process_log("query_received", {"query": query})
    
    def set_configuration(self, config: Dict[str, Any]):
        """Log the workflow configuration."""
        self.log_data["configuration"] = {
            "mode": config.get("mode", "simple"),
            "llm_model": config.get("llm_model", "unknown"),
            "enable_offline_rag": config.get("enable_offline_rag", False),
            "enable_courtlistener": config.get("enable_courtlistener", False),
            "max_iterations": config.get("max_iterations", 3)
        }
        self._add_process_log("configuration_set", self.log_data["configuration"])
    
    def log_search_query(self, query_type: str, query: str, priority: str = "high"):
        """Log a search query that will be executed."""
        search_query = {
            "timestamp": datetime.now().isoformat(),
            "query_type": query_type,
            "query": query,
            "priority": priority
        }
        self.log_data["search_queries"].append(search_query)
        self._add_process_log("search_query_generated", search_query)
    
    def log_search_result(self, source: str, results: List[Dict[str, Any]], query: str = ""):
        """
        Log search results with COMPLETE content (no truncation).
        
        Args:
            source: Source of the search (offline_rag, web_search, case_law)
            results: List of search results
            query: The query that produced these results
        """
        for result in results:
            # Ensure we capture FULL content without truncation
            search_result = {
                "timestamp": datetime.now().isoformat(),
                "source": source,
                "query": query,
                "title": result.get("title", ""),
                "content": result.get("content", ""),  # Full content
                "full_content": result.get("full_content", result.get("content", "")),  # Ensure we have complete content
                "url": result.get("url", ""),
                "file_path": result.get("file_path", ""),
                "relevance_score": result.get("relevance_score", 0),
                "confidence": result.get("confidence", 0),
                "chunk_id": result.get("chunk_id", ""),
                "metadata": result.get("metadata", {})
            }
            
            # Store complete result
            self.log_data["search_results"].append(search_result)
            self.log_data["metadata"]["total_search_results"] += 1
            
            # Add source to metadata if not already there
            if source not in self.log_data["metadata"]["sources_used"]:
                self.log_data["metadata"]["sources_used"].append(source)
        
        self._add_process_log(f"search_completed_{source}", {
            "source": source,
            "results_count": len(results),
            "query": query
        })
    
    def log_llm_interaction(self, agent_name: str, input_prompt: str, output_response: Any, 
                           model: str = "", metadata: Dict[str, Any] = None):
        """
        Log LLM interactions with COMPLETE prompts and responses.
        
        Args:
            agent_name: Name of the agent (query_agent, judge_agent, etc.)
            input_prompt: Complete input prompt sent to LLM
            output_response: Complete response from LLM
            model: Model used for this interaction
            metadata: Additional metadata
        """
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "agent": agent_name,
            "model": model,
            "input_prompt": input_prompt,  # Full prompt, no truncation
            "output_response": str(output_response),  # Full response
            "metadata": metadata or {}
        }
        
        self.log_data["llm_interactions"].append(interaction)
        self._add_process_log(f"llm_interaction_{agent_name}", {
            "agent": agent_name,
            "prompt_length": len(input_prompt),
            "response_length": len(str(output_response))
        })
    
    def log_final_answer(self, answer: Dict[str, Any]):
        """Log the final answer with complete content."""
        self.log_data["final_answer"] = {
            "timestamp": datetime.now().isoformat(),
            "answer": answer.get("answer", ""),  # Full answer
            "key_points": answer.get("key_points", []),
            "sources": answer.get("sources", []),
            "confidence": answer.get("confidence", 0),
            "disclaimers": answer.get("disclaimers", [])
        }
        self._add_process_log("final_answer_generated", {
            "answer_length": len(answer.get("answer", "")),
            "key_points_count": len(answer.get("key_points", [])),
            "sources_count": len(answer.get("sources", []))
        })
    
    def log_error(self, error_type: str, error_message: str, context: Dict[str, Any] = None):
        """Log any errors that occur during processing."""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": error_type,
            "message": error_message,
            "context": context or {}
        }
        self.log_data["metadata"]["errors"].append(error_entry)
        self._add_process_log(f"error_{error_type}", error_entry)
    
    def log_follow_up_questions(self, questions: List[Dict[str, str]]):
        """Log follow-up questions generated by the system."""
        self.log_data["follow_up_questions"] = [
            {
                "question": q.get("question", ""),
                "reason": q.get("reason", "")
            }
            for q in questions
        ]
        self._add_process_log("follow_up_questions_generated", {
            "count": len(questions)
        })
    
    def log_user_responses(self, responses: Dict[str, str]):
        """Log user responses to follow-up questions."""
        self.log_data["user_responses"] = responses
        self._add_process_log("user_responses_received", {
            "responses_count": len(responses)
        })
    
    def _add_process_log(self, event: str, data: Any = None):
        """Add an entry to the process log."""
        self.log_data["process_log"].append({
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "data": data
        })
    
    def save(self):
        """Save the complete log to file."""
        # Calculate duration
        duration = (datetime.now() - self.start_time).total_seconds()
        self.log_data["metadata"]["duration_seconds"] = duration
        
        # Add completion timestamp
        self.log_data["completed_at"] = datetime.now().isoformat()
        
        # Write to file with proper formatting
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(self.log_data, f, indent=2, ensure_ascii=False)
        
        self._add_process_log("log_saved", {"file": str(self.log_file)})
        
        return str(self.log_file)
    
    def get_log_summary(self) -> Dict[str, Any]:
        """Get a summary of the current log."""
        return {
            "session_id": self.session_id,
            "query": self.log_data["query"],
            "total_search_results": self.log_data["metadata"]["total_search_results"],
            "sources_used": self.log_data["metadata"]["sources_used"],
            "llm_interactions_count": len(self.log_data["llm_interactions"]),
            "errors_count": len(self.log_data["metadata"]["errors"]),
            "log_file": str(self.log_file)
        }


# Singleton instance for current session
_current_logger: Optional[ResultLogger] = None


def get_logger() -> Optional[ResultLogger]:
    """Get the current session logger."""
    return _current_logger


def create_logger(results_dir: str = "results", session_id: str = None) -> ResultLogger:
    """Create a new logger for the session."""
    global _current_logger
    _current_logger = ResultLogger(results_dir, session_id)
    return _current_logger


def close_logger():
    """Close and save the current logger."""
    global _current_logger
    if _current_logger:
        log_file = _current_logger.save()
        _current_logger = None
        return log_file
    return None