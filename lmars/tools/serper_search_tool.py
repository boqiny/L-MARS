"""
Google Serper Search tool for LegalAgents.
Adapted and modified from https://github.com/sunnynexus/Search-o1/blob/main/scripts/bing_search.py

WORKFLOW 1: Basic Search (search_serper_web)
-------------------------------------------
User calls: search_serper_web("immigration law", max_results=5)
    ↓
1. serper_web_search() - Makes API call to Google Serper
    ↓
2. extract_relevant_info() - Parses JSON response into structured data
    ↓ 
3. Returns formatted results (titles, URLs, snippets only)

WORKFLOW 2: Enhanced Search with Content (search_serper_with_content) 
--------------------------------------------------------------------
User calls: search_serper_with_content("F1 visa rules", max_results=3)
    ↓
1. serper_web_search() - Makes API call to Google Serper
    ↓
2. extract_relevant_info() - Parses JSON response 
    ↓
3. For each URL:
       extract_text_from_url() - Scrapes webpage content
           ↓
       IF snippet provided:
           extract_snippet_with_context() - Finds best matching content
               ↓
           Uses helper functions:
               - remove_punctuation() - Cleans text
               - f1_score() - Calculates text similarity
               - sent_tokenize() - Splits into sentences
       ↓
4. Returns formatted results with full webpage content

USAGE:
======
# Quick search - just titles and snippets
results = search_serper_web("trademark law")

# Deep search - full webpage content  
detailed_results = search_serper_with_content("patent infringement cases")
"""
import os
import json
import requests
from requests.exceptions import Timeout
from bs4 import BeautifulSoup
import pdfplumber
from io import BytesIO
import re
import string
from typing import Optional, Tuple, Dict, List

try:
    from nltk.tokenize import sent_tokenize
except ImportError:
    # Fallback sentence tokenizer if NLTK is not available
    def sent_tokenize(text):
        return re.split(r'(?<=[.!?])\s+', text)


# Custom headers for web scraping
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/58.0.3029.110 Safari/537.36',
    'Referer': 'https://www.google.com/',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}

# Initialize session
session = requests.Session()
session.headers.update(headers)


# =============================================================================
# MAIN ENTRY POINTS - These are the functions users call
# =============================================================================

def search_serper_web(query: str, max_results: int = 5) -> str:
    """
    BASIC SEARCH: Get search results with titles, URLs, and snippets only.
    Fast and lightweight - no content extraction from individual pages.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return
        
    Returns:
        Formatted string containing search results with extracted content
    """
    try:
        # Get Serper API credentials
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            return "Error: SERPER_API_KEY environment variable not set. Please add your Serper API key."
        
        # Step 1: Make API call
        search_results = serper_web_search(query, api_key)
        
        if not search_results:
            return "No search results found or API request failed."
        
        # Step 2: Parse response into structured format
        extracted_info = extract_relevant_info(search_results)
        
        if not extracted_info:
            return "No relevant information found in search results."
        
        # Step 3: Limit and format results
        extracted_info = extracted_info[:max_results]
        
        # Format results for output
        formatted_results = []
        formatted_results.append(f"Found {len(extracted_info)} relevant web results:\n")
        
        for info in extracted_info:
            result_text = []
            result_text.append(f"{info['id']}. {info['title']}")
            result_text.append(f"   URL: {info['url']}")
            if info.get('site_name'):
                result_text.append(f"   Site: {info['site_name']}")
            if info.get('date'):
                result_text.append(f"   Date: {info['date']}")
            if info.get('snippet'):
                result_text.append(f"   Summary: {info['snippet']}")
            
            formatted_results.append("\n".join(result_text))
            formatted_results.append("")
        
        return "\n".join(formatted_results)
        
    except Exception as e:
        return f"Error during Serper search: {str(e)}"


def search_serper_with_content(query: str, max_results: int = 3) -> str:
    """
    ENHANCED SEARCH: Get search results AND extract full content from each webpage.
    Slower but more comprehensive - scrapes actual webpage content.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to process for content extraction
        
    Returns:
        Formatted string containing search results with extracted content
    """
    try:
        # Get Serper API credentials
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            return "Error: SERPER_API_KEY environment variable not set."
        
        # Step 1: Make API call (same as basic search)
        search_results = serper_web_search(query, api_key)
        
        if not search_results:
            return "No search results found or API request failed."
        
        # Step 2: Parse response (same as basic search)
        extracted_info = extract_relevant_info(search_results)
        
        if not extracted_info:
            return "No relevant information found in search results."
        
        # Step 3: Limit results for content extraction (fewer because it's slower)
        extracted_info = extracted_info[:max_results]
        
        # Step 4: Extract content from each URL (this is the enhancement)
        for info in extracted_info:
            try:
                full_text = extract_text_from_url(info['url'], snippet=info.get('snippet'))
                if full_text and not full_text.startswith("Error"):
                    info['content'] = full_text[:2000]  # Limit content length
                else:
                    info['content'] = "Could not extract content from this page"
            except Exception as e:
                info['content'] = f"Error extracting content: {str(e)}"
        
        # Step 5: Format results with content
        formatted_results = []
        formatted_results.append(f"Found {len(extracted_info)} web results with content:\n")
        
        for info in extracted_info:
            result_text = []
            result_text.append(f"{info['id']}. {info['title']}")
            result_text.append(f"   URL: {info['url']}")
            if info.get('site_name'):
                result_text.append(f"   Site: {info['site_name']}")
            if info.get('snippet'):
                result_text.append(f"   Summary: {info['snippet']}")
            if info.get('content'):
                result_text.append(f"   Content: {info['content']}")
            
            formatted_results.append("\n".join(result_text))
            formatted_results.append("")
        
        return "\n".join(formatted_results)
        
    except Exception as e:
        return f"Error during Serper search with content: {str(e)}"


# =============================================================================
# CORE API INTERFACE
# =============================================================================

def serper_web_search(query: str, api_key: str, timeout: int = 20) -> Dict:
    """
    CORE API CALL: Makes the actual HTTP request to Google Serper API.
    This is where we interface with the external service.
    """
    url = "https://google.serper.dev/search"
    
    payload = json.dumps({
        "q": query
    })
    
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(url, data=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
        search_results = response.json()
        return search_results
    except Timeout:
        print(f"Serper API request timed out ({timeout} seconds) for query: {query}")
        return {}
    except requests.exceptions.RequestException as e:
        print(f"Error occurred during Serper API request: {e}")
        return {}


def extract_relevant_info(search_results: Dict) -> List[Dict]:
    """
    RESPONSE PARSER: Converts Serper's JSON response into our standardized format.
    Handles both organic results and knowledge graph data.
    """
    useful_info = []
    
    # Extract from organic results (main search results)
    if 'organic' in search_results:
        for id, result in enumerate(search_results['organic']):
            # Extract site name from URL
            site_name = ""
            try:
                from urllib.parse import urlparse
                parsed_url = urlparse(result.get('link', ''))
                site_name = parsed_url.netloc.replace('www.', '')
            except:
                pass
            
            info = {
                'id': id + 1,
                'title': result.get('title', ''),
                'url': result.get('link', ''),
                'site_name': site_name,
                'date': result.get('date', ''),
                'snippet': result.get('snippet', ''),
                'content': ''
            }
            useful_info.append(info)
    
    # Also include knowledge graph if available
    if 'knowledgeGraph' in search_results:
        kg = search_results['knowledgeGraph']
        info = {
            'id': len(useful_info) + 1,
            'title': kg.get('title', '') + " (Knowledge Graph)",
            'url': kg.get('descriptionLink', ''),
            'site_name': 'Google Knowledge Graph',
            'date': '',
            'snippet': kg.get('description', ''),
            'content': ''
        }
        useful_info.append(info)
    
    return useful_info


# =============================================================================
# CONTENT EXTRACTION SYSTEM
# =============================================================================

def extract_text_from_url(url, snippet: Optional[str] = None):
    """
    WEB SCRAPER: Downloads and extracts text from web pages and PDFs.
    Uses snippet context to find the most relevant parts of long pages.
    """
    try:
        response = session.get(url, timeout=20)
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', '')
        if 'pdf' in content_type:
            return extract_pdf_text(url)
        
        try:
            soup = BeautifulSoup(response.text, 'lxml')
        except Exception:
            soup = BeautifulSoup(response.text, 'html.parser')
        
        text = soup.get_text(separator=' ', strip=True)

        if snippet:
            # Try to find the most relevant section based on search snippet
            success, context = extract_snippet_with_context(text, snippet)
            if success:
                return context
            else:
                return text
        else:
            return text[:8000]
            
    except requests.exceptions.HTTPError as http_err:
        return f"HTTP error occurred: {http_err}"
    except requests.exceptions.ConnectionError:
        return "Error: Connection error occurred"
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Unexpected error: {str(e)}"


def extract_snippet_with_context(full_text: str, snippet: str, context_chars: int = 2500) -> Tuple[bool, str]:
    """
    SMART CONTENT FINDER: Locates the sentence that best matches the search snippet
    and returns it with surrounding context for better relevance.
    """
    try:
        full_text = full_text[:50000]
        snippet = snippet.lower()
        snippet = remove_punctuation(snippet)
        snippet_words = set(snippet.split())

        best_sentence = None
        best_f1 = 0.2

        sentences = sent_tokenize(full_text)

        # Find the sentence most similar to our search snippet
        for sentence in sentences:
            key_sentence = sentence.lower()
            key_sentence = remove_punctuation(key_sentence)
            sentence_words = set(key_sentence.split())
            f1 = f1_score(snippet_words, sentence_words)
            if f1 > best_f1:
                best_f1 = f1
                best_sentence = sentence

        if best_sentence:
            # Extract context around the best matching sentence
            para_start = full_text.find(best_sentence)
            para_end = para_start + len(best_sentence)
            start_index = max(0, para_start - context_chars)
            end_index = min(len(full_text), para_end + context_chars)
            context = full_text[start_index:end_index]
            return True, context
        else:
            return False, full_text[:context_chars * 2]
    except Exception as e:
        return False, f"Failed to extract snippet context due to {str(e)}"


def extract_pdf_text(url):
    """PDF EXTRACTOR: Specialized handler for PDF documents."""
    try:
        response = session.get(url, timeout=20)
        if response.status_code != 200:
            return f"Error: Unable to retrieve the PDF (status code {response.status_code})"
        
        with pdfplumber.open(BytesIO(response.content)) as pdf:
            full_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text
        
        cleaned_text = ' '.join(full_text.split()[:600])
        return cleaned_text
    except requests.exceptions.Timeout:
        return "Error: Request timed out after 20 seconds"
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# TEXT PROCESSING UTILITIES
# =============================================================================

def remove_punctuation(text: str) -> str:
    """Remove punctuation from the text."""
    return text.translate(str.maketrans("", "", string.punctuation))


def f1_score(true_set: set, pred_set: set) -> float:
    """Calculate the F1 score between two sets of words."""
    intersection = len(true_set.intersection(pred_set))
    if not intersection:
        return 0.0
    precision = intersection / float(len(pred_set))
    recall = intersection / float(len(true_set))
    return 2 * (precision * recall) / (precision + recall)


if __name__ == "__main__":
    """Test the Serper search tool"""
    
    print("=" * 70)
    print("GOOGLE SERPER SEARCH TOOL")
    print("=" * 70)
    from dotenv import load_dotenv
    load_dotenv()
    
    # Test queries for different scenarios
    test_queries = [
        "can i work remotely in the united states as f1 student",
        # "employment law remote work regulations"
    ]
    
    # Check API key
    api_key = os.getenv("SERPER_API_KEY")
    if api_key:
        print("✓ Serper API key found")
    else:
        print("⚠ No Serper API key found - set SERPER_API_KEY environment variable")
        exit(1)
    
    for i, test_query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}: {test_query}")
        print(f"{'='*70}")
        
        try:
            # Test 1: Basic Search (Fast)
            print(f"\n📋 BASIC SEARCH (Titles & Snippets Only)")
            print("-" * 50)
            print("⏱️  Should be fast, lightweight results...")
            
            basic_result = search_serper_web(test_query, max_results=3)
            print("BASIC RESULTS:")
            print(basic_result)
            
            # Test 2: Enhanced Search (With Content)
            print(f"\n📄 ENHANCED SEARCH (Full Content Extraction)")
            print("-" * 50)
            print("⏱️  Slower but more comprehensive, scraping actual webpage content...")
            
            enhanced_result = search_serper_with_content(test_query, max_results=2)
            print("ENHANCED RESULTS:")
            # Show more content for enhanced search to demonstrate the difference
            if len(enhanced_result) > 2000:
                print(enhanced_result[:2000] + f"\n... [Content truncated for display - Total length: {len(enhanced_result)} chars]")
            else:
                print(enhanced_result)
            
            # Analysis section
            print(f"\n📊 COMPARISON ANALYSIS:")
            print("-" * 30)
            basic_lines = basic_result.count('\n')
            enhanced_lines = enhanced_result.count('\n')
            print(f"• Basic search output:    {len(basic_result):,} characters, {basic_lines} lines")
            print(f"• Enhanced search output: {len(enhanced_result):,} characters, {enhanced_lines} lines")
            print(f"• Content enhancement:    {len(enhanced_result) - len(basic_result):,} additional characters")
            
            # Check for content extraction indicators
            has_content = "Content:" in enhanced_result
            has_extraction_errors = "Error extracting content" in enhanced_result
            print(f"• Content successfully extracted: {'✓' if has_content and not has_extraction_errors else '✗'}")
            
            if i < len(test_queries):
                print(f"\n⏳ Moving to next test query...")
                
        except Exception as e:
            print(f"❌ Error in test {i}: {e}")
            continue