"""
Multi-agent system for legal question answering with structured output.
Each agent has a specific role and uses Pydantic models for structured communication.
"""
from typing import List, Optional, Literal, Union
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage
import uuid

# Structured output models
class FollowUpQuestion(BaseModel):
    """A follow-up question to clarify the user's legal query."""
    question: str = Field(description="The clarifying question to ask")
    reason: str = Field(description="Why this question helps provide better assistance")

class QueryGeneration(BaseModel):
    """Generated search query for legal research."""
    query: str = Field(description="Specific search query for legal databases or web search")
    query_type: Literal["case_law", "web_search"] = Field(description="Type of search to perform")
    priority: Literal["high", "medium", "low"] = Field(description="Priority level for this query")

class SearchResult(BaseModel):
    """Structured search result from legal databases or web search."""
    source: str = Field(description="Source of the information (e.g., CourtListener, Serper web search)")
    title: str = Field(description="Title or case name")
    content: str = Field(description="Relevant content excerpt")
    url: Optional[str] = Field(description="URL to full source", default=None)
    confidence: float = Field(description="Confidence score 0-1", ge=0, le=1)

class JudgmentResult(BaseModel):
    """Judge agent's evaluation of search results."""
    is_sufficient: bool = Field(description="Whether the results sufficiently answer the question")
    missing_information: List[str] = Field(description="What information is still needed", default=[])
    suggested_refinements: List[str] = Field(description="Suggested query refinements", default=[])
    confidence: float = Field(description="Confidence in the judgment", ge=0, le=1)

class FinalAnswer(BaseModel):
    """Final structured answer to the legal question."""
    answer: str = Field(description="Comprehensive answer to the legal question")
    key_points: List[str] = Field(description="Key legal points and considerations")
    sources: List[str] = Field(description="Sources used in the answer")
    confidence: float = Field(description="Overall confidence in the answer", ge=0, le=1)
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
    confidence: float = Field(description="Confidence in the analysis", ge=0, le=1)

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
        - contract: For contract-related queries
        
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
        8. Confidence in your analysis
        
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
            'contract_generation': tools.get('contract')
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
            
        elif query.query_type == "contract" and self.tools['contract_generation']:
            # Use contract tool for contract-related queries
            raw_results = self.tools['contract_generation'](query.query)
            results.extend(self._parse_contract_results(raw_results, query.query))
        
        return results
    
    def _parse_serper_results(self, raw_results: str, query: str) -> List[SearchResult]:
        """Parse Serper search results into structured format."""
        # Basic parsing - in practice, you'd implement more sophisticated parsing
        return [SearchResult(
            source="Serper Web Search",
            title=f"Search results for: {query}",
            content=raw_results[:500] + "..." if len(raw_results) > 500 else raw_results,
            confidence=0.7
        )]
    
    def _parse_courtlistener_results(self, raw_results: str, query: str) -> List[SearchResult]:
        """Parse CourtListener results into structured format."""
        return [SearchResult(
            source="CourtListener",
            title=f"Legal cases for: {query}",
            content=raw_results[:500] + "..." if len(raw_results) > 500 else raw_results,
            confidence=0.8
        )]
    
    def _parse_contract_results(self, raw_results: str, query: str) -> List[SearchResult]:
        """Parse contract tool results into structured format."""
        return [SearchResult(
            source="Contract Generation Tool",
            title=f"Contract information for: {query}",
            content=raw_results[:500] + "..." if len(raw_results) > 500 else raw_results,
            confidence=0.6
        )]

class JudgeAgent:
    """Agent responsible for evaluating if search results answer the user's question."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def evaluate_results(self, user_query: str, search_results: List[SearchResult]) -> JudgmentResult:
        """Evaluate if the search results sufficiently answer the user's question."""
        
        structured_llm = self.llm.with_structured_output(JudgmentResult)
        
        results_summary = "\n".join([
            f"Source: {r.source}\nTitle: {r.title}\nContent: {r.content[:200]}...\n"
            for r in search_results
        ])
        
        prompt = f"""
        Evaluate whether the search results sufficiently answer the user's legal question.
        
        User's Question: {user_query}
        
        Search Results:
        {results_summary}
        
        Analyze:
        1. Do the results directly address the question?
        2. Is the information comprehensive enough?
        3. What key information might be missing?
        4. Should we refine the search with different queries?
        
        Be thorough but practical in your evaluation.
        """
        
        response = structured_llm.invoke([HumanMessage(content=prompt)])
        return response

class SummaryAgent:
    """Agent responsible for creating the final answer from search results."""
    
    def __init__(self, llm):
        self.llm = llm
    
    def generate_final_answer(self, user_query: str, search_results: List[SearchResult]) -> FinalAnswer:
        """Generate a comprehensive final answer based on search results."""
        
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
        4. High confidence in your response based on available information
        
        Remember: This is informational only and not legal advice.
        """
        
        response = structured_llm.invoke([HumanMessage(content=prompt)])
        
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