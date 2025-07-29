"""
L-MARS: LangGraph-based multi-agent workflow for legal question answering.
Implements orchestrated reasoning and agentic search similar to OpenAI's DeepResearch with legal domain focus.
"""
from typing import Annotated, Dict, Any, List
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
import os

from .agents import (
    QueryAgent, SearchAgent, JudgeAgent, SummaryAgent,
    FollowUpQuestion, QueryGeneration, SearchResult, JudgmentResult, FinalAnswer
)
from .tools.serper_search_tool import search_serper_with_content
from .tools.courtlistener_tool import find_legal_cases


class LMarsState(TypedDict):
    """State for the L-MARS multi-agent workflow."""
    messages: Annotated[List[BaseMessage], add_messages]
    original_query: str
    follow_up_questions: List[FollowUpQuestion]
    user_responses: Dict[str, str]
    search_queries: List[QueryGeneration]
    search_results: List[SearchResult]
    judgment: JudgmentResult
    final_answer: FinalAnswer
    current_step: str
    iteration_count: int
    max_iterations: int


class LMarsGraph:
    """L-MARS: Multi-agent legal research workflow using LangGraph."""
    
    def __init__(self, llm_model: str = "openai:gpt-4", max_iterations: int = 3):
        self.llm = init_chat_model(llm_model)
        self.max_iterations = max_iterations
        
        # Initialize agents
        self.query_agent = QueryAgent(self.llm)
        self.judge_agent = JudgeAgent(self.llm)
        self.summary_agent = SummaryAgent(self.llm)
        
        # Initialize tools for search agent
        tools = {
            'serper': search_serper_with_content,
            'courtlistener': find_legal_cases,
            'contract': None  # Contract tool is minimal, skip for now
        }
        self.search_agent = SearchAgent(tools)
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        # Create the state graph
        graph_builder = StateGraph(LMarsState)
        
        # Add nodes
        graph_builder.add_node("query_processing", self._query_processing_node)
        graph_builder.add_node("follow_up_questions", self._follow_up_questions_node)
        graph_builder.add_node("generate_queries", self._generate_queries_node)
        graph_builder.add_node("search_execution", self._search_execution_node)
        graph_builder.add_node("judge_evaluation", self._judge_evaluation_node)
        graph_builder.add_node("generate_summary", self._generate_summary_node)
        
        # Add edges and conditional routing
        graph_builder.add_edge(START, "query_processing")
        graph_builder.add_edge("query_processing", "follow_up_questions")
        
        # Conditional edge: if follow-up questions needed, wait for user input
        graph_builder.add_conditional_edges(
            "follow_up_questions",
            self._should_ask_followup,
            {
                "ask_user": END,  # Pause for user input
                "continue": "generate_queries"
            }
        )
        
        graph_builder.add_edge("generate_queries", "search_execution")
        graph_builder.add_edge("search_execution", "judge_evaluation")
        
        # Conditional edge: judge decides if we need more search or can summarize
        graph_builder.add_conditional_edges(
            "judge_evaluation",
            self._should_continue_search,
            {
                "search_more": "generate_queries",
                "summarize": "generate_summary"
            }
        )
        
        graph_builder.add_edge("generate_summary", END)
        
        # Compile with memory for state persistence
        memory = InMemorySaver()
        return graph_builder.compile(checkpointer=memory)
    
    def _query_processing_node(self, state: LMarsState) -> Dict[str, Any]:
        """Initial processing of the user's query."""
        
        # Extract the user's query from messages
        last_message = state["messages"][-1] if state["messages"] else None
        if last_message and hasattr(last_message, 'content'):
            query = last_message.content
        else:
            query = "No query provided"
        
        return {
            "original_query": query,
            "current_step": "query_processing",
            "iteration_count": 0,
            "max_iterations": self.max_iterations
        }
    
    def _follow_up_questions_node(self, state: LMarsState) -> Dict[str, Any]:
        """Generate follow-up questions to clarify the user's needs."""
        
        # Generate follow-up questions
        questions = self.query_agent.generate_followup_questions(
            state["original_query"], 
            state["messages"]
        )
        
        # Add AI message with follow-up questions
        question_text = "I'd like to ask a few clarifying questions to better help you:\n\n"
        for i, q in enumerate(questions, 1):
            question_text += f"{i}. {q.question}\n   (This helps because: {q.reason})\n\n"
        
        ai_message = AIMessage(content=question_text)
        
        return {
            "follow_up_questions": questions,
            "messages": [ai_message],
            "current_step": "follow_up_questions"
        }
    
    def _should_ask_followup(self, state: LMarsState) -> str:
        """Decide if we should ask follow-up questions or continue."""
        
        # If we have follow-up questions and no responses yet, ask user
        if state.get("follow_up_questions") and not state.get("user_responses"):
            return "ask_user"
        
        return "continue"
    
    def _generate_queries_node(self, state: LMarsState) -> Dict[str, Any]:
        """Generate specific search queries based on refined understanding."""
        
        # Build context from user responses
        context = ""
        if state.get("user_responses"):
            context = " | ".join([f"{k}: {v}" for k, v in state["user_responses"].items()])
        
        # Generate search queries
        queries = self.query_agent.generate_search_queries(
            state["original_query"], 
            context
        )
        
        return {
            "search_queries": queries,
            "current_step": "generate_queries"
        }
    
    def _search_execution_node(self, state: LMarsState) -> Dict[str, Any]:
        """Execute searches using available tools."""
        
        all_results = []
        
        # Execute each search query
        for query in state.get("search_queries", []):
            try:
                results = self.search_agent.execute_search(query)
                all_results.extend(results)
            except Exception as e:
                # Log error and continue with other searches
                print(f"Search error for query '{query.query}': {e}")
                continue
        
        return {
            "search_results": all_results,
            "current_step": "search_execution"
        }
    
    def _judge_evaluation_node(self, state: LMarsState) -> Dict[str, Any]:
        """Evaluate if search results are sufficient."""
        
        judgment = self.judge_agent.evaluate_results(
            state["original_query"],
            state.get("search_results", [])
        )
        
        return {
            "judgment": judgment,
            "current_step": "judge_evaluation"
        }
    
    def _should_continue_search(self, state: LMarsState) -> str:
        """Decide if we need more search or can generate final answer."""
        
        judgment = state.get("judgment")
        iteration_count = state.get("iteration_count", 0)
        max_iterations = state.get("max_iterations", 3)
        
        # If judgment says insufficient and we haven't hit max iterations
        if (judgment and not judgment.is_sufficient and 
            iteration_count < max_iterations):
            return "search_more"
        
        return "summarize"
    
    def _generate_summary_node(self, state: LMarsState) -> Dict[str, Any]:
        """Generate the final answer for the user."""
        
        final_answer = self.summary_agent.generate_final_answer(
            state["original_query"],
            state.get("search_results", [])
        )
        
        # Create summary message
        summary_text = f"## Legal Research Summary\n\n"
        summary_text += f"**Question:** {state['original_query']}\n\n"
        summary_text += f"**Answer:** {final_answer.answer}\n\n"
        
        if final_answer.key_points:
            summary_text += "**Key Points:**\n"
            for point in final_answer.key_points:
                summary_text += f"• {point}\n"
            summary_text += "\n"
        
        if final_answer.sources:
            summary_text += "**Sources:**\n"
            for source in final_answer.sources:
                summary_text += f"• {source}\n"
            summary_text += "\n"
        
        if final_answer.disclaimers:
            summary_text += "**Important Disclaimers:**\n"
            for disclaimer in final_answer.disclaimers:
                summary_text += f"• {disclaimer}\n"
        
        ai_message = AIMessage(content=summary_text)
        
        return {
            "final_answer": final_answer,
            "messages": [ai_message],
            "current_step": "complete"
        }
    
    def invoke(self, user_input: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Main entry point for the legal research system."""
        
        if config is None:
            config = {"configurable": {"thread_id": "legal_session"}}
        
        # Create initial state
        initial_state = {
            "messages": [HumanMessage(content=user_input)]
        }
        
        # Run the graph
        result = self.graph.invoke(initial_state, config)
        return result
    
    def stream(self, user_input: str, config: Dict[str, Any] = None):
        """Stream the legal research process."""
        
        if config is None:
            config = {"configurable": {"thread_id": "legal_session"}}
        
        initial_state = {
            "messages": [HumanMessage(content=user_input)]
        }
        
        for event in self.graph.stream(initial_state, config, stream_mode="values"):
            yield event
    
    def continue_conversation(self, user_responses: Dict[str, str], config: Dict[str, Any] = None):
        """Continue the conversation after user provides follow-up answers."""
        
        if config is None:
            config = {"configurable": {"thread_id": "legal_session"}}
        
        # Update state with user responses
        current_state = self.graph.get_state(config)
        updated_state = {
            **current_state.values,
            "user_responses": user_responses,
            "messages": current_state.values.get("messages", []) + 
                       [HumanMessage(content=f"User responses: {user_responses}")]
        }
        
        # Continue from generate_queries
        result = self.graph.invoke(updated_state, config)
        return result


def create_legal_mind_graph(llm_model: str = "openai:gpt-4") -> LMarsGraph:
    """Factory function to create an L-MARS graph instance."""
    return LMarsGraph(llm_model=llm_model)


# Example usage
if __name__ == "__main__":
    # Initialize the system
    lmars = create_legal_mind_graph()
    
    # Example query
    user_query = "Can an F1 student work remotely for a US company while studying?"
    
    print("L-MARS Research System Starting...")
    print(f"Query: {user_query}")
    print("-" * 50)
    
    # Stream the process
    config = {"configurable": {"thread_id": "demo_session"}}
    
    for event in lmars.stream(user_query, config):
        current_step = event.get("current_step", "unknown")
        print(f"Step: {current_step}")
        
        if "messages" in event and event["messages"]:
            last_message = event["messages"][-1]
            if hasattr(last_message, 'content'):
                print(f"Output: {last_message.content[:200]}...")
        
        print("-" * 30)