"""
CourtListener API tool for comprehensive legal search.
Refer to official API documentation: https://www.courtlistener.com/help/api/rest/search/
"""
import os
import requests
from typing import Dict, Any, Optional


class CourtListenerSearch:
    """Simple CourtListener API client for legal case searches."""
    
    def __init__(self, api_token: Optional[str] = None):
        """Initialize the CourtListener search client."""
        self.base_url = "https://www.courtlistener.com/api/rest/v4"
        self.api_token = api_token or os.getenv("COURTLISTENER_API_TOKEN")
        self.headers = {}
        if self.api_token:
            self.headers["Authorization"] = f"Token {self.api_token}"
    
    def search_cases(self, 
                    query: str,
                    limit: int = 10,
                    court: Optional[str] = None,
                    date_range: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Search for legal cases relevant to a user's question.
        
        Args:
            query: User's legal question or search terms
            limit: Maximum results to return (1-20)
            court: Filter by specific court (optional)
            date_range: Date filtering {'start': 'YYYY-MM-DD', 'end': 'YYYY-MM-DD'}
            
        Returns:
            Dictionary with search results and metadata
        """
        
        # Build query parameters for case law search
        params = {
            "q": query,
            "type": "o",  # Case law opinions only
            "page_size": min(max(limit, 1), 20),
            "order_by": "score desc",  # Most relevant first
            "highlight": "on"
        }
        
        # Add court filter if specified
        if court:
            params["court"] = court
        
        # Add date range filter if specified
        if date_range:
            if 'start' in date_range:
                params["filed_after"] = date_range['start']
            if 'end' in date_range:
                params["filed_before"] = date_range['end']
        
        try:
            response = requests.get(
                f"{self.base_url}/search/",
                params=params,
                headers=self.headers,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "error": f"API request failed with status {response.status_code}",
                    "details": response.text[:200] if response.text else "No details available"
                }
                
        except requests.exceptions.RequestException as e:
            return {"error": f"Request failed: {str(e)}"}


def search_legal_cases(query: str, 
                      limit: int = 10,
                      court: Optional[str] = None) -> str:
    """
    Search for legal cases to help answer a user's legal question.
    
    Args:
        query: User's legal question or search terms
        limit: Maximum number of results (default: 10, max: 20)
        court: Filter by court (e.g., 'scotus' for Supreme Court)
        
    Returns:
        Formatted string with relevant legal cases
    """
    
    client = CourtListenerSearch()
    results = client.search_cases(
        query=query,
        limit=limit,
        court=court
    )
    
    if "error" in results:
        return f"Search Error: {results['error']}"
    
    return _format_case_results(results, query)


def _format_case_results(results: Dict[str, Any], query: str) -> str:
    """Format search results for legal case information."""
    
    if results.get("count", 0) == 0:
        return f"No legal cases found for: '{query}'"
    
    count = results["count"]
    cases = results.get("results", [])
    
    # Format header
    output = []
    output.append("=" * 60)
    output.append(f"LEGAL CASES FOR: {query}")
    output.append(f"Found: {count:,} cases (showing top {len(cases)})")
    output.append("=" * 60)
    
    # Format each case
    for i, case in enumerate(cases, 1):
        case_info = []
        
        # Case name
        case_name = case.get("caseName", "Unknown Case")
        case_info.append(f"\n{i}. {case_name}")
        
        # Court and date
        court = case.get("court", "Unknown Court")
        date_filed = case.get("dateFiled", "Unknown Date")
        case_info.append(f"   Court: {court} | Date: {date_filed}")
        
        # Legal authority (how often cited)
        cite_count = case.get("citeCount", 0)
        if cite_count > 0:
            case_info.append(f"   Authority: Cited by {cite_count} other cases")
        
        # Key excerpt with search terms highlighted
        opinions = case.get("opinions", [])
        if opinions and opinions[0].get("snippet"):
            snippet = opinions[0]["snippet"].strip()
            # Clean up snippet length
            if len(snippet) > 250:
                snippet = snippet[:250] + "..."
            case_info.append(f"   Key text: {snippet}")
        
        # Link to full case
        abs_url = case.get("absolute_url")
        if abs_url:
            case_info.append(f"   Link: https://www.courtlistener.com{abs_url}")
        
        output.extend(case_info)
        if i < len(cases):
            output.append("")  # Space between cases
    
    return "\n".join(output)


# Main function for easy use
def find_legal_cases(user_question: str, max_results: int = 5) -> str:
    """
    Find legal cases relevant to a user's legal question.
    
    Args:
        user_question: The user's legal question
        max_results: Number of cases to return (default: 5)
        
    Returns:
        Formatted legal cases that help answer the question
    """
    return search_legal_cases(user_question, limit=max_results)


if __name__ == "__main__":
    """Simple test of the legal case search."""
    
    print("Testing Legal Case Search Tool")
    print("=" * 40)
    
    # Test with a sample legal question
    test_question = "Can F1 students work in the US?"
    print(f"Question: {test_question}")
    print("\nSearching for relevant cases...\n")
    
    try:
        result = find_legal_cases(test_question, max_results=3)
        print(result)
    except Exception as e:
        print(f"Error: {e}")