"""
L-MARS Workflow Module - Simplified with Two Modes
Simple Mode: Single-turn retrieval and response
Multi-Turn Mode: Iterative refinement with agents
"""
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
import uuid

from .agents import SearchAgent, SummaryAgent, QueryAgent, JudgeAgent
from .agents import SearchResult, FinalAnswer, QueryGeneration
from .tools.serper_search_tool import search_serper_web, search_serper_with_content
from .tools.courtlistener_tool import find_legal_cases
from .tools.offline_rag_tool import search_offline_rag
from .trajectory_tracker import TrajectoryTracker
from .result_logger import get_logger
from .evaluation import LegalAnswerEvaluator
from .llm_judge import CombinedEvaluator


class WorkflowConfig(BaseModel):
    """Configuration for L-MARS workflow."""
    mode: Literal["simple", "multi_turn"] = Field(default="simple", description="Workflow mode")
    llm_model: str = Field(default="openai:gpt-4o", description="Main LLM model")
    judge_model: Optional[str] = Field(default=None, description="Judge model for multi-turn mode")
    max_iterations: int = Field(default=3, description="Max iterations for multi-turn mode")
    enable_tracking: bool = Field(default=True, description="Enable trajectory tracking")
    # Search source configuration (online search is always enabled by default)
    enable_offline_rag: bool = Field(default=False, description="Enable offline RAG search from local documents")
    enable_courtlistener: bool = Field(default=False, description="Enable CourtListener case law search")


class SimpleWorkflow:
    """Simple single-turn workflow for legal research."""
    
    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.llm = init_chat_model(config.llm_model)
        self.tracker = TrajectoryTracker() if config.enable_tracking else None
        
        # Initialize tools
        tools = {
            'serper': search_serper_web,  # Use basic search for faster multiple results
            'courtlistener': find_legal_cases,
            'offline_rag': search_offline_rag,
        }
        self.search_agent = SearchAgent(tools)
        self.summary_agent = SummaryAgent(self.llm)
    
    def run(self, query: str) -> Dict[str, Any]:
        """Execute simple workflow: search -> summarize."""
        run_id = None
        logger = get_logger()
        
        if self.tracker:
            run_id = self.tracker.start_run(query, {"mode": "simple", "llm_model": str(self.llm)})
        
        # Log query
        if logger:
            logger.set_query(query)
            logger.set_configuration({
                "mode": "simple",
                "llm_model": str(self.llm),
                "enable_offline_rag": self.config.enable_offline_rag,
                "enable_courtlistener": self.config.enable_courtlistener
            })
        
        try:
            # Generate search queries directly
            search_queries = self._generate_search_queries(query)
            
            # Log search queries
            if logger:
                for sq in search_queries:
                    logger.log_search_query(sq.query_type, sq.query, sq.priority)
            
            # Execute searches
            search_results = self._execute_searches(search_queries)
            
            # Generate final answer
            final_answer = self.summary_agent.generate_final_answer(query, search_results)
            
            # Combined evaluation (quantitative + qualitative)
            combined_evaluator = CombinedEvaluator()
            combined_evaluation = combined_evaluator.evaluate(
                answer_text=final_answer.answer,
                sources=final_answer.sources,
                jurisdiction=None,  # Could extract from query
                question=query,
                search_results=[{
                    'source': r.source,
                    'content': r.content
                } for r in search_results] if search_results else None
            )
            
            # Extract quantitative metrics for backward compatibility
            evaluation_metrics = combined_evaluation['quantitative']
            
            # Log final answer with both evaluations
            if logger and final_answer:
                eval_data = {
                    "answer": final_answer.answer,
                    "key_points": final_answer.key_points,
                    "sources": final_answer.sources,
                    "disclaimers": final_answer.disclaimers,
                    "evaluation": {
                        "u_score": evaluation_metrics.u_score,
                        "hedging_score": evaluation_metrics.hedging_score,
                        "temporal_vagueness": evaluation_metrics.temporal_vagueness,
                        "citation_score": evaluation_metrics.citation_score,
                        "jurisdiction_score": evaluation_metrics.jurisdiction_score,
                        "decisiveness_score": evaluation_metrics.decisiveness_score
                    }
                }
                
                # Add qualitative evaluation if available
                if combined_evaluation.get('qualitative'):
                    qual = combined_evaluation['qualitative']
                    eval_data['qualitative_evaluation'] = {
                        'factual_accuracy': qual.factual_accuracy.level,
                        'evidence_grounding': qual.evidence_grounding.level,
                        'clarity_reasoning': qual.clarity_reasoning.level,
                        'uncertainty_awareness': qual.uncertainty_awareness.level,
                        'overall_usefulness': qual.overall_usefulness.level,
                        'summary': qual.summary
                    }
                
                logger.log_final_answer(eval_data)
            
            result = {
                "query": query,
                "search_results": search_results,
                "final_answer": final_answer,
                "evaluation_metrics": evaluation_metrics,
                "combined_evaluation": combined_evaluation,
                "mode": "simple"
            }
            
            if self.tracker:
                self.tracker.end_run(result)
            
            return result
            
        except Exception as e:
            if logger:
                logger.log_error("workflow_error", str(e))
            if self.tracker:
                self.tracker.end_run({"error": str(e)})
            raise
    
    def _generate_search_queries(self, query: str) -> List[QueryGeneration]:
        """Generate search queries based on enabled sources."""
        queries = []
        
        # Always include web search (online search)
        queries.append(
            QueryGeneration(
                query=query,
                query_type="web_search",
                priority="high"
            )
        )
        
        # Only add offline RAG if explicitly enabled
        if self.config.enable_offline_rag:
            queries.insert(0,  # Put offline RAG first when enabled
                QueryGeneration(
                    query=query,
                    query_type="offline_rag",
                    priority="high"
                )
            )
        
        # Only add CourtListener if explicitly enabled
        if self.config.enable_courtlistener:
            queries.append(
                QueryGeneration(
                    query=query,
                    query_type="case_law",
                    priority="medium"
                )
            )
        
        return queries
    
    def _execute_searches(self, queries: List[QueryGeneration]) -> List[SearchResult]:
        """Execute searches and return results."""
        all_results = []
        sources_tried = []
        sources_succeeded = []
        logger = get_logger()
        
        for query in queries:
            source_name = query.query_type.replace("_", " ").title()
            sources_tried.append(source_name)
            
            print(f"  📍 Searching {source_name}...", end="")
            
            try:
                results = self.search_agent.execute_search(query)
                if results:
                    all_results.extend(results)
                    sources_succeeded.append(source_name)
                    print(f" ✓ Found {len(results)} result(s)")
                    
                    # Log complete search results
                    if logger:
                        # Convert SearchResult objects to dicts with full content
                        results_data = []
                        for r in results:
                            if hasattr(r, '__dict__'):
                                result_dict = r.__dict__
                            else:
                                result_dict = {
                                    "source": r.source if hasattr(r, 'source') else "",
                                    "title": r.title if hasattr(r, 'title') else "",
                                    "content": r.content if hasattr(r, 'content') else "",
                                    "url": r.url if hasattr(r, 'url') else ""
                                }
                            results_data.append(result_dict)
                        
                        logger.log_search_result(query.query_type, results_data, query.query)
                else:
                    print(f" (no results)")
            except Exception as e:
                # Log errors
                if logger:
                    logger.log_error(f"search_error_{query.query_type}", str(e), 
                                   {"query": query.query})
                
                # Silently skip if API key missing or service unavailable
                if "api_key" in str(e).lower() or "API" in str(e):
                    print(f" (skipped - no API key)")
                    continue
                else:
                    print(f" ✗ Error")
                    if "verbose" in str(self.config):
                        print(f"    Details: {e}")
                continue
        
        # Summary
        print(f"\n📊 Search complete: {len(all_results)} total results from {len(sources_succeeded)} source(s)")
        
        return all_results


class MultiTurnWorkflow:
    """Multi-turn workflow with agent refinement."""
    
    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.llm = init_chat_model(config.llm_model)
        
        # Judge model can be different
        judge_model = config.judge_model or config.llm_model
        if "o3-mini" in judge_model:
            self.judge_llm = init_chat_model(judge_model, temperature=0, model_kwargs={"parallel_tool_calls": False})
        else:
            self.judge_llm = init_chat_model(judge_model, temperature=0)  # Temperature=0 for reproducible judge evaluations
        
        self.tracker = TrajectoryTracker() if config.enable_tracking else None
        
        # Initialize agents
        self.query_agent = QueryAgent(self.llm)
        self.judge_agent = JudgeAgent(self.judge_llm)
        self.summary_agent = SummaryAgent(self.llm)
        
        # Initialize tools
        tools = {
            'serper': search_serper_with_content,  # Use deep search for thorough analysis in multi-turn mode
            'courtlistener': find_legal_cases,
            'offline_rag': search_offline_rag,
        }
        self.search_agent = SearchAgent(tools)
    
    def run(self, query: str, user_responses: Dict[str, str] = None) -> Dict[str, Any]:
        """Execute multi-turn workflow with refinement loop."""
        run_id = None
        logger = get_logger()
        
        if self.tracker:
            run_id = self.tracker.start_run(query, {
                "mode": "multi_turn",
                "llm_model": str(self.llm),
                "max_iterations": self.config.max_iterations
            })
        
        # Log query and configuration
        if logger:
            logger.set_query(query)
            logger.set_configuration({
                "mode": "multi_turn",
                "llm_model": str(self.llm),
                "judge_model": str(self.judge_llm),
                "max_iterations": self.config.max_iterations,
                "enable_offline_rag": self.config.enable_offline_rag,
                "enable_courtlistener": self.config.enable_courtlistener
            })
        
        try:
            # Step 1: Generate follow-up questions
            follow_up_questions = self.query_agent.generate_followup_questions(query, [])
            
            # If no user responses provided, return questions for user input
            if not user_responses and follow_up_questions:
                return {
                    "query": query,
                    "follow_up_questions": follow_up_questions,
                    "needs_user_input": True,
                    "mode": "multi_turn"
                }
            
            # Step 2: Generate refined search queries
            context = ""
            if user_responses:
                context = " | ".join([f"{k}: {v}" for k, v in user_responses.items()])
            
            # Generate queries respecting enabled sources
            all_queries = self.query_agent.generate_search_queries(query, context)
            
            # Filter based on enabled sources
            search_queries = []
            for q in all_queries:
                if q.query_type == "web_search":
                    search_queries.append(q)  # Always include web search
                elif q.query_type == "offline_rag" and self.config.enable_offline_rag:
                    search_queries.append(q)
                elif q.query_type == "case_law" and self.config.enable_courtlistener:
                    search_queries.append(q)
            
            # Step 3: Iterative search and refinement
            all_search_results = []
            all_executed_queries = set()  # Track all queries to avoid duplicates
            iteration = 0
            
            while iteration < self.config.max_iterations:
                # Execute searches
                print(f"\n🔄 Search Iteration {iteration + 1}")
                
                # Filter out duplicate queries
                unique_queries = []
                for query_obj in search_queries:
                    query_text = query_obj.query.lower().strip()
                    if query_text not in all_executed_queries:
                        unique_queries.append(query_obj)
                        all_executed_queries.add(query_text)
                
                if not unique_queries:
                    print("  ℹ️  No new queries to execute (all would be duplicates)")
                    if iteration > 0:  # Force break if we've already done searches
                        break
                    unique_queries = search_queries  # On first iteration, use all queries
                
                print(f"📍 Executing {len(unique_queries)} search queries...")
                
                iteration_results = []
                for query_obj in unique_queries:
                    print(f"  • Searching {query_obj.query_type}: \"{query_obj.query[:60]}...\"" if len(query_obj.query) > 60 else f"  • Searching {query_obj.query_type}: \"{query_obj.query}\"")
                    try:
                        results = self.search_agent.execute_search(query_obj)
                        if results:
                            iteration_results.extend(results)
                            print(f"    ✓ Found {len(results)} result(s)")
                        else:
                            print(f"    - No results")
                    except Exception as e:
                        print(f"    ✗ Error: {str(e)[:100]}")
                        continue
                
                all_search_results.extend(iteration_results)
                print(f"📊 Total results so far: {len(all_search_results)}")
                
                # Judge evaluation
                judgment = self.judge_agent.evaluate_results(
                    query,
                    all_search_results,
                    iteration_count=iteration
                )
                
                # Log judge evaluation
                if logger:
                    logger.log_llm_interaction(
                        agent_name=f"judge_agent_iteration_{iteration + 1}",
                        input_prompt=f"Evaluating {len(all_search_results)} search results",
                        output_response={
                            "is_sufficient": judgment.is_sufficient,
                            "reasoning": judgment.reasoning,
                            "source_quality": judgment.source_quality,
                            "date_check": judgment.date_check,
                            "jurisdiction_check": judgment.jurisdiction_check,
                            "contradiction_check": judgment.contradiction_check,
                            "missing_information": judgment.missing_information,
                            "suggested_refinements": judgment.suggested_refinements
                        },
                        model=str(self.judge_llm),
                        metadata={"iteration": iteration + 1}
                    )
                
                # Print judge evaluation (verbose output)
                print(f"\n🔍 JUDGE EVALUATION (Iteration {iteration + 1}):")
                print("-" * 60)
                print(f"✅ Sufficient: {judgment.is_sufficient}")
                
                if judgment.reasoning:
                    print(f"\n📝 Chain-of-Thought Reasoning:")
                    print(f"   {judgment.reasoning[:300]}..." if len(judgment.reasoning) > 300 else f"   {judgment.reasoning}")
                
                if judgment.source_quality:
                    print(f"\n📊 Source Quality Analysis:")
                    print(f"   {judgment.source_quality}")
                
                if judgment.date_check:
                    print(f"\n📅 Date Relevance Check:")
                    print(f"   {judgment.date_check}")
                
                if judgment.jurisdiction_check:
                    print(f"\n⚖️ Jurisdiction Check:")
                    print(f"   {judgment.jurisdiction_check}")
                
                if judgment.contradiction_check:
                    print(f"\n⚠️ Contradiction Analysis:")
                    print(f"   {judgment.contradiction_check}")
                
                if judgment.missing_information:
                    print(f"\n❓ Missing Information:")
                    for info in judgment.missing_information[:3]:  # Show max 3
                        print(f"   • {info}")
                
                if judgment.suggested_refinements:
                    print(f"\n💡 Suggested Refinements:")
                    for refinement in judgment.suggested_refinements[:3]:  # Show max 3
                        print(f"   • {refinement}")
                
                print("-" * 60)
                
                # If sufficient, break
                if judgment.is_sufficient:
                    print("✅ Judge determined results are SUFFICIENT. Proceeding to final answer.")
                    break
                else:
                    print(f"🔄 Judge determined results are INSUFFICIENT. Continuing search...")
                    print()
                
                # Generate new queries based on judge feedback
                if judgment.suggested_refinements:
                    # Use the judge's suggested refinements directly as search queries
                    search_queries = []
                    for refinement in judgment.suggested_refinements[:2]:  # Limit to 2 new queries
                        # Clean up the refinement to make it a better search query
                        # Remove phrases like "Search for", "Look for", "Find" at the beginning
                        clean_query = refinement
                        for prefix in ["Search for ", "Look for ", "Find ", "Seek ", "Search ", "Inquire about "]:
                            if clean_query.startswith(prefix):
                                clean_query = clean_query[len(prefix):]
                                break
                        
                        # Remove quotes that make searches too specific
                        clean_query = clean_query.replace('"', '').replace("'", "")
                        
                        search_queries.append(
                            QueryGeneration(
                                query=clean_query,
                                query_type="web_search",
                                priority="high"
                            )
                        )
                elif judgment.missing_information:
                    # Create specific search queries for missing information
                    search_queries = []
                    for missing_info in judgment.missing_information[:2]:  # Limit to 2 new queries
                        # Create a focused search query for the missing information
                        # Combine context from original query with specific missing info
                        search_queries.append(
                            QueryGeneration(
                                query=missing_info,
                                query_type="web_search",  
                                priority="high"
                            )
                        )
                
                iteration += 1
            
            # Step 4: Generate final answer
            final_answer = self.summary_agent.generate_final_answer(query, all_search_results)
            
            # Combined evaluation (quantitative + qualitative)
            combined_evaluator = CombinedEvaluator()
            combined_evaluation = combined_evaluator.evaluate(
                answer_text=final_answer.answer,
                sources=final_answer.sources,
                jurisdiction=None,  # Could extract from user_responses
                question=query,
                search_results=[{
                    'source': r.source,
                    'content': r.content
                } for r in all_search_results] if all_search_results else None
            )
            
            # Extract quantitative metrics for backward compatibility
            evaluation_metrics = combined_evaluation['quantitative']
            
            # Log final answer with both evaluations
            if logger and final_answer:
                eval_data = {
                    "answer": final_answer.answer,
                    "key_points": final_answer.key_points,
                    "sources": final_answer.sources,
                    "disclaimers": final_answer.disclaimers,
                    "evaluation": {
                        "u_score": evaluation_metrics.u_score,
                        "hedging_score": evaluation_metrics.hedging_score,
                        "temporal_vagueness": evaluation_metrics.temporal_vagueness,
                        "citation_score": evaluation_metrics.citation_score,
                        "jurisdiction_score": evaluation_metrics.jurisdiction_score,
                        "decisiveness_score": evaluation_metrics.decisiveness_score
                    }
                }
                
                # Add qualitative evaluation if available
                if combined_evaluation.get('qualitative'):
                    qual = combined_evaluation['qualitative']
                    eval_data['qualitative_evaluation'] = {
                        'factual_accuracy': qual.factual_accuracy.level,
                        'evidence_grounding': qual.evidence_grounding.level,
                        'clarity_reasoning': qual.clarity_reasoning.level,
                        'uncertainty_awareness': qual.uncertainty_awareness.level,
                        'overall_usefulness': qual.overall_usefulness.level,
                        'summary': qual.summary
                    }
                
                logger.log_final_answer(eval_data)
            
            result = {
                "query": query,
                "user_responses": user_responses,
                "search_results": all_search_results,
                "final_answer": final_answer,
                "evaluation_metrics": evaluation_metrics,
                "combined_evaluation": combined_evaluation,
                "iterations": iteration + 1,
                "mode": "multi_turn"
            }
            
            if self.tracker:
                self.tracker.end_run(result)
            
            return result
            
        except Exception as e:
            if self.tracker:
                self.tracker.end_run({"error": str(e)})
            raise


class LMarsWorkflow:
    """Main L-MARS workflow orchestrator with mode selection."""
    
    def __init__(self, config: WorkflowConfig = None):
        self.config = config or WorkflowConfig()
        
        # Initialize appropriate workflow based on mode
        if self.config.mode == "simple":
            self.workflow = SimpleWorkflow(self.config)
        else:
            self.workflow = MultiTurnWorkflow(self.config)
    
    def run(self, query: str, user_responses: Dict[str, str] = None) -> Dict[str, Any]:
        """Execute workflow based on configured mode."""
        if isinstance(self.workflow, SimpleWorkflow):
            return self.workflow.run(query)
        else:
            return self.workflow.run(query, user_responses)
    
    def set_mode(self, mode: Literal["simple", "multi_turn"]):
        """Switch between workflow modes."""
        self.config.mode = mode
        if mode == "simple":
            self.workflow = SimpleWorkflow(self.config)
        else:
            self.workflow = MultiTurnWorkflow(self.config)


def create_workflow(
    mode: Literal["simple", "multi_turn"] = "simple",
    llm_model: str = "openai:gpt-4o",
    judge_model: str = None,
    max_iterations: int = 3,
    enable_tracking: bool = True,
    enable_offline_rag: bool = False,
    enable_courtlistener: bool = False
) -> LMarsWorkflow:
    """Factory function to create L-MARS workflow."""
    config = WorkflowConfig(
        mode=mode,
        llm_model=llm_model,
        judge_model=judge_model,
        max_iterations=max_iterations,
        enable_tracking=enable_tracking,
        enable_offline_rag=enable_offline_rag,
        enable_courtlistener=enable_courtlistener
    )
    return LMarsWorkflow(config)