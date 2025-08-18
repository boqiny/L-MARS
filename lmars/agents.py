"""
Multi-agent system for legal question answering with structured output.
Each agent has a specific role and uses Pydantic models for structured communication.
"""
from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage
import uuid
from .result_logger import get_logger

# Structured output models
class FollowUpQuestion(BaseModel):
    """A follow-up question to clarify the user's legal query."""
    question: str = Field(description="The clarifying question to ask")
    reason: str = Field(description="Why this question helps provide better assistance")

class QueryGeneration(BaseModel):
    """Generated search query for legal research."""
    query: str = Field(description="Specific search query for legal databases or web search")
    query_type: Literal["case_law", "web_search", "offline_rag"] = Field(description="Type of search to perform")
    priority: Literal["high", "medium", "low"] = Field(description="Priority level for this query")

class SearchResult(BaseModel):
    """Structured search result from legal databases or web search."""
    source: str = Field(description="Source of the information (e.g., CourtListener, Serper web search)")
    title: str = Field(description="Title or case name")
    content: str = Field(description="Relevant content excerpt")
    url: Optional[str] = Field(description="URL to full source", default=None)

class JudgmentResult(BaseModel):
    """Judge agent's evaluation of search results."""
    is_sufficient: bool = Field(description="Whether the results sufficiently answer the question")
    reasoning: str = Field(description="Chain of thought reasoning for the judgment", default="")
    source_quality: str = Field(description="Analysis of source quality (gov/court/edu vs other)", default="")
    date_check: str = Field(description="Analysis of date relevance and currency", default="")
    jurisdiction_check: str = Field(description="Analysis of jurisdiction match", default="")
    contradiction_check: str = Field(description="Check for contradictions in sources", default="")
    missing_information: List[str] = Field(description="What information is still needed", default=[])
    suggested_refinements: List[str] = Field(description="Suggested query refinements", default=[])

class FinalAnswer(BaseModel):
    """Final structured answer to the legal question."""
    answer: str = Field(description="Comprehensive answer to the legal question")
    key_points: List[str] = Field(description="Key legal points and considerations")
    sources: List[str] = Field(description="Sources used in the answer")
    disclaimers: List[str] = Field(description="Important legal disclaimers", default=[])

class Person(BaseModel):
    """Information about a person involved in the legal matter."""
    name: Optional[str] = Field(description="Person's name if provided", default=None)
    role: str = Field(description="Role in the legal matter (e.g., plaintiff, defendant, witness, client)")
    background: Optional[str] = Field(description="Relevant background information", default=None)
    relationship: Optional[str] = Field(description="Relationship to the user/case", default=None)

class QueryResult(BaseModel):
    """Structured result from query analysis with detailed categorization."""
    id: str = Field(description="Unique identifier for this query result", default_factory=lambda: str(uuid.uuid4()))
    category: str = Field(description="Legal category/area (e.g., contract law, criminal law, family law)")
    people: List[Person] = Field(description="People involved in the legal matter", default=[])
    jurisdiction: Optional[str] = Field(description="Relevant jurisdiction if specified", default=None)
    urgency: Literal["high", "medium", "low"] = Field(description="Urgency level of the legal matter")
    legal_areas: List[str] = Field(description="Specific areas of law involved", default=[])
    timeline: Optional[str] = Field(description="Relevant timeline or deadlines", default=None)
    context: str = Field(description="Additional context about the legal situation")

# Wrapper classes for List types to fix structured output issues
class FollowUpQuestionList(BaseModel):
    """Wrapper for list of follow-up questions."""
    questions: List[FollowUpQuestion] = Field(description="List of follow-up questions")

class QueryGenerationList(BaseModel):
    """Wrapper for list of query generations."""
    queries: List[QueryGeneration] = Field(description="List of generated queries")

# Agent implementations
class QueryAgent:
    """Agent responsible for understanding user queries and generating follow-up questions."""
    
    def __init__(self, llm, structured_output: bool = False):
        self.llm = llm
        self.structured_output = structured_output
    
    def generate_followup_questions(self, user_query: str, conversation_history: List[BaseMessage]) -> List[FollowUpQuestion]:
        """Generate follow-up questions to clarify the user's legal needs."""
        
        structured_llm = self.llm.with_structured_output(FollowUpQuestionList)
        
        prompt = f"""
        You are a legal assistant helping users with their legal questions. 
        
        User's original question: {user_query}
        
        Generate 2-3 clarifying follow-up questions that would help you provide better legal assistance.
        Focus on:
        - Jurisdiction/location if not specified
        - Specific circumstances or context
        - Timeline or urgency
        - What type of help they need (information, next steps, etc.)
        
        Only ask questions that would significantly help in providing better assistance.
        """
        
        response = structured_llm.invoke([HumanMessage(content=prompt)])
        return response.questions
    
    def generate_search_queries(self, user_query: str, context: str = "") -> List[QueryGeneration]:
        """Generate specific search queries based on the user's refined question."""
        
        structured_llm = self.llm.with_structured_output(QueryGenerationList)
        
        prompt = f"""
        Based on the user's legal question and any additional context, generate 2-4 specific search queries.
        
        User Question: {user_query}
        Additional Context: {context}
        
        Generate queries for different sources:
        - case_law: For searching legal cases and precedents
        - web_search: For general legal information and recent updates
        - offline_rag: For searching local legal documents in the inputs folder
        
        Make queries specific and focused to get the most relevant results.
        """
        
        response = structured_llm.invoke([HumanMessage(content=prompt)])
        return response.queries
    
    def analyze_query(self, user_query: str, context: str = "") -> Union[str, QueryResult]:
        """Analyze user query and return either plain text or structured result based on configuration."""
        
        if self.structured_output:
            return self._generate_structured_query_result(user_query, context)
        else:
            return self._generate_plain_text_analysis(user_query, context)
    
    def _generate_structured_query_result(self, user_query: str, context: str = "") -> QueryResult:
        """Generate structured query result with detailed categorization."""
        
        structured_llm = self.llm.with_structured_output(QueryResult)
        
        prompt = f"""
        Analyze the user's legal query and provide a detailed structured breakdown.
        
        User Query: {user_query}
        Additional Context: {context}
        
        Analyze and extract:
        1. Legal category/area (e.g., contract law, criminal law, family law, employment law)
        2. People involved with their roles (plaintiff, defendant, client, witness, etc.)
        3. Jurisdiction if mentioned or inferable
        4. Urgency level based on the nature of the query
        5. Specific legal areas involved
        6. Timeline or deadlines if mentioned
        7. Additional context about the situation
        
        Be thorough but accurate in your categorization.
        """
        
        response = structured_llm.invoke([HumanMessage(content=prompt)])
        return response
    
    def _generate_plain_text_analysis(self, user_query: str, context: str = "") -> str:
        """Generate plain text analysis of the user query."""
        
        prompt = f"""
        Analyze the user's legal query and provide a brief text summary of the key aspects.
        
        User Query: {user_query}
        Additional Context: {context}
        
        Provide a concise analysis covering:
        - Legal area/category
        - Key parties involved
        - Urgency level
        - Important context
        """
        
        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

class SearchAgent:
    """Agent responsible for executing searches using available tools."""
    
    def __init__(self, tools):
        self.tools = {
            'serper_search': tools.get('serper'),
            'courtlistener': tools.get('courtlistener'),
            'offline_rag': tools.get('offline_rag')
        }
    
    def execute_search(self, query: QueryGeneration) -> List[SearchResult]:
        """Execute search based on query type and return structured results."""
        
        results = []
        
        if query.query_type == "web_search" and self.tools['serper_search']:
            # Use Serper for web search
            raw_results = self.tools['serper_search'](query.query)
            results.extend(self._parse_serper_results(raw_results, query.query))
            
        elif query.query_type == "case_law" and self.tools['courtlistener']:
            # Use CourtListener for case law
            raw_results = self.tools['courtlistener'](query.query)
            results.extend(self._parse_courtlistener_results(raw_results, query.query))
            
        elif query.query_type == "offline_rag" and self.tools['offline_rag']:
            # Use offline RAG for local documents
            raw_results = self.tools['offline_rag'](query.query)
            results.extend(self._parse_offline_rag_results(raw_results, query.query))
        
        return results
    
    def _parse_serper_results(self, raw_results: str, query: str) -> List[SearchResult]:
        """Parse Serper search results into structured format."""
        results = []
        
        # Parse the formatted Serper output
        # Formats: 
        # Basic: "Found X relevant web results:\n\n1. Title\n   URL: ...\n   Site: ...\n   Summary: ...\n\n2. ..."
        # With content: "Found X web results with content:\n\n1. Title\n   URL: ...\n   Site: ...\n   Summary: ...\n   Content: ...\n\n2. ..."
        lines = raw_results.split('\n')
        
        current_result = None
        current_title = ""
        current_url = ""
        current_site = ""
        current_summary = ""
        current_content = ""
        
        for line in lines:
            line = line.strip()
            
            # Check if this is a numbered result (e.g., "1. Title")
            if line and line[0].isdigit() and '. ' in line:
                # Save previous result if exists
                if current_title and (current_summary or current_content):
                    # Use content if available (from deep search), otherwise use summary
                    final_content = current_content if current_content else current_summary
                    results.append(SearchResult(
                        source=f"Web Search - {current_site}" if current_site else "Web Search",
                        title=current_title,
                        content=final_content,
                        url=current_url if current_url else None
                    ))
                
                # Start new result
                current_title = line.split('. ', 1)[1] if '. ' in line else line
                current_url = ""
                current_site = ""
                current_summary = ""
                current_content = ""
                
            elif line.startswith('URL: '):
                current_url = line[5:]
            elif line.startswith('Site: '):
                current_site = line[6:]
            elif line.startswith('Summary: '):
                current_summary = line[9:]
            elif line.startswith('Content: '):
                # For deep search results with full content
                current_content = line[9:]
            elif line.startswith('Date: '):
                # Include date in summary if present
                if current_summary:
                    current_summary = f"[{line[6:]}] {current_summary}"
        
        # Add the last result
        if current_title and (current_summary or current_content):
            final_content = current_content if current_content else current_summary
            results.append(SearchResult(
                source=f"Web Search - {current_site}" if current_site else "Web Search",
                title=current_title,
                content=final_content,
                url=current_url if current_url else None
            ))
        
        # If no results were parsed, fall back to treating it as a single result
        if not results:
            results.append(SearchResult(
                source="Web Search",
                title=f"Search results for: {query}",
                content=raw_results[:1000] + "..." if len(raw_results) > 1000 else raw_results
            ))
        
        return results
    
    def _parse_courtlistener_results(self, raw_results: str, query: str) -> List[SearchResult]:
        """Parse CourtListener results into structured format."""
        return [SearchResult(
            source="CourtListener",
            title=f"Legal cases for: {query}",
            content=raw_results[:500] + "..." if len(raw_results) > 500 else raw_results
        )]
    
    def _parse_offline_rag_results(self, raw_results: str, query: str) -> List[SearchResult]:
        """Parse offline RAG results into structured format."""
        return [SearchResult(
            source="Offline RAG (Local Documents)",
            title=f"Local legal documents for: {query}",
            content=raw_results[:1000] + "..." if len(raw_results) > 1000 else raw_results
        )]

class JudgeAgent:
    """Agent responsible for evaluating if search results answer the user's question."""
    
    def __init__(self, llm):
        self.llm = llm
        self.previous_evaluations = []
    
    def evaluate_results(self, user_query: str, search_results: List[SearchResult], 
                        conversation_history: List[BaseMessage] = None, 
                        iteration_count: int = 0) -> JudgmentResult:
        """Evaluate if the search results sufficiently answer the user's question."""
        
        try:
            structured_llm = self.llm.with_structured_output(JudgmentResult)
        except Exception as e:
            # Fallback for models that don't support structured output properly
            structured_llm = self.llm
        
        results_summary = "\n".join([
            f"Source: {r.source}\nTitle: {r.title}\nContent: {r.content}\n"
            for r in search_results
        ])
        
        # Build conversation context
        conversation_context = ""
        if conversation_history:
            conversation_context = "\n".join([
                f"{'Human' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
                for msg in conversation_history[-10:]  # Last 10 messages for context
            ])
        
        # Build previous evaluation context
        prev_evaluations = ""
        if self.previous_evaluations:
            prev_evaluations = "\nPrevious Evaluations:\n" + "\n".join([
                f"Iteration {i}: {'Sufficient' if eval_result.is_sufficient else 'Insufficient'} - Missing: {', '.join(eval_result.missing_information)}"
                for i, eval_result in enumerate(self.previous_evaluations[-3:], 1)  # Last 3 evaluations
            ])
        
        prompt = f"""
        You are a legal research judge evaluating search results. This is iteration {iteration_count + 1}.
        
        Original Question: {user_query}
        
        Conversation History:
        {conversation_context}
        
        Current Search Results ({len(search_results)} results):
        {results_summary}
        {prev_evaluations}
        
        EVALUATION PROTOCOL WITH CHAIN-OF-THOUGHT:
        
        1. REASONING (Chain of Thought):
        Think step by step about whether the search results answer the user's specific question.
        Consider: What was asked? What information was provided? What is still missing?
        
        2. SOURCE QUALITY CHECK:
        Analyze source authority - are there sources from:
        - Government (.gov) sites? 
        - Court decisions/legal databases?
        - Educational institutions (.edu)?
        - How many authoritative vs user-generated content sources?
        
        3. DATE CHECK:
        - Are the sources current and relevant to today's date?
        - If there are older sources, do we also have recent confirmations?
        - Flag if critical information might be outdated
        
        4. JURISDICTION CHECK:
        - Does the jurisdiction of sources match the user's location/scope?
        - For US federal vs state law, is the distinction clear?
        - User mentioned: {conversation_context if conversation_context else "No specific jurisdiction mentioned"}
        
        5. CONTRADICTION SCAN:
        - Do any sources contradict each other?
        - If yes, what specific elements conflict?
        - Do we need more specific queries to resolve conflicts?
        
        STOP RULE:
        Mark as SUFFICIENT when:
        - Primary claims have authoritative support (gov/court/edu when available)
        - No critical information gaps for practical guidance
        - No unresolved contradictions
        - Jurisdiction and dates are appropriate
        
        Mark as INSUFFICIENT when:
        - Missing critical legal requirements or procedures
        - Only user-generated content for key claims
        - Significant contradictions need resolution
        - Wrong jurisdiction or outdated information
        
        IMPORTANT for iteration {iteration_count + 1}:
        - Be MORE LENIENT after multiple iterations
        - Focus on whether user has enough info to take action
        - Consider cumulative information across all iterations
        
        SUGGESTED REFINEMENTS (if insufficient):
        Provide 1-2 specific search queries that would help find the missing information.
        These should be actual search queries that can be executed directly, NOT instructions.
        Good examples:
        - "California SB 365 employment arbitration 2025 full text"
        - "Biden student loan forgiveness Supreme Court ruling 2024"
        - "AB 465 arbitration agreement requirements California"
        Bad examples (don't do this):
        - "Search for more information about SB 365"
        - "Look for court rulings"
        - "Find government sources"
        
        Provide detailed reasoning for your judgment.
        """
        
        try:
            response = structured_llm.invoke([HumanMessage(content=prompt)])
            
            # If we got a structured response, use it directly
            if isinstance(response, JudgmentResult):
                result = response
            else:
                # Parse unstructured response
                content = response.content if hasattr(response, 'content') else str(response)
                result = self._parse_judgment_response(content)
            
        except Exception as e:
            print(f"Warning: Judge evaluation failed: {e}")
            # Return a default response
            result = JudgmentResult(
                is_sufficient=iteration_count >= 2,  # Be lenient after 2 iterations
                missing_information=["Could not evaluate due to model error"],
                suggested_refinements=[]
            )
        
        # Store this evaluation for future reference
        self.previous_evaluations.append(result)
        
        return result
    
    def _parse_judgment_response(self, content: str) -> JudgmentResult:
        """Parse unstructured response into JudgmentResult."""
        # Simple parsing logic - could be improved
        is_sufficient = "sufficient" in content.lower() and "insufficient" not in content.lower()
        
        return JudgmentResult(
            is_sufficient=is_sufficient,
            reasoning=content[:500] if len(content) > 500 else content,
            source_quality="Unable to parse source quality from response",
            date_check="Unable to parse date check from response",
            jurisdiction_check="Unable to parse jurisdiction check from response",
            contradiction_check="Unable to parse contradiction check from response",
            missing_information=[] if is_sufficient else ["Additional information needed"],
            suggested_refinements=[] if is_sufficient else ["Refine search queries"]
        )

class SummaryAgent:
    """Agent responsible for creating the final answer from search results."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def generate_final_answer(self, user_query: str, search_results: List[SearchResult]) -> FinalAnswer:
        """Generate a comprehensive final answer based on search results."""
        logger = get_logger()
        
        structured_llm = self.llm.with_structured_output(FinalAnswer)
        
        results_content = "\n\n".join([
            f"From {r.source}:\n{r.content}"
            for r in search_results
        ])
        
        sources = list(set([r.source for r in search_results]))
        
        prompt = f"""
        Create a comprehensive answer to the user's legal question based on the search results.
        
        User's Question: {user_query}
        
        Search Results:
        {results_content}
        
        Provide:
        1. A clear, comprehensive answer
        2. Key legal points and considerations
        3. Important disclaimers about legal advice
        
        Remember: This is informational only and not legal advice.
        """
        
        # Log LLM interaction
        if logger:
            logger.log_llm_interaction(
                agent_name="summary_agent",
                input_prompt=prompt,
                output_response="[Generating final answer...]",
                model=str(self.llm),
                metadata={"search_results_count": len(search_results)}
            )
        
        response = structured_llm.invoke([HumanMessage(content=prompt)])
        
        # Log the actual response
        if logger:
            logger.log_llm_interaction(
                agent_name="summary_agent_response",
                input_prompt=prompt,
                output_response=str(response),
                model=str(self.llm),
                metadata={"final_answer": True}
            )
        
        # Ensure sources are included
        if not response.sources:
            response.sources = sources
        
        # Add standard legal disclaimers
        standard_disclaimers = [
            "This information is for educational purposes only and does not constitute legal advice.",
            "Consult with a qualified attorney for advice specific to your situation.",
            "Laws vary by jurisdiction and may have changed since this information was compiled."
        ]
        
        response.disclaimers.extend(standard_disclaimers)
        
        return response