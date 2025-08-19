"""
Rate limit configuration and utilities for evaluation scripts.
Helps avoid 429 errors when running batch evaluations.
"""
import time
import re
from typing import Optional, Callable, Any


class RateLimitHandler:
    """Handles rate limiting for API calls."""
    
    def __init__(self, base_delay: float = 3.0, max_retries: int = 3):
        """
        Initialize rate limit handler.
        
        Args:
            base_delay: Base delay in seconds when rate limited
            max_retries: Maximum number of retries
        """
        self.base_delay = base_delay
        self.max_retries = max_retries
        self.last_request_time = 0
        self.min_request_interval = 0.1  # Minimum time between requests
        
    def wait_if_needed(self):
        """Ensure minimum time between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def extract_wait_time(self, error_str: str) -> float:
        """Extract wait time from rate limit error message."""
        wait_match = re.search(r'try again in ([\d.]+)s', error_str)
        if wait_match:
            return float(wait_match.group(1)) + 0.5  # Add buffer
        return self.base_delay
    
    def is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error is a rate limit error."""
        error_str = str(error)
        return '429' in error_str or 'rate_limit' in error_str.lower()
    
    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function with automatic retry on rate limit.
        
        Args:
            func: Function to execute
            *args: Positional arguments for func
            **kwargs: Keyword arguments for func
            
        Returns:
            Result from func
            
        Raises:
            Exception: If max retries exceeded or non-rate-limit error
        """
        for attempt in range(self.max_retries):
            try:
                self.wait_if_needed()
                return func(*args, **kwargs)
            except Exception as e:
                if self.is_rate_limit_error(e):
                    wait_time = self.extract_wait_time(str(e))
                    if attempt < self.max_retries - 1:
                        print(f"\n⚠️  Rate limited. Waiting {wait_time:.1f}s before retry {attempt + 2}/{self.max_retries}...")
                        time.sleep(wait_time)
                    else:
                        print(f"\n❌ Rate limit exceeded after {self.max_retries} attempts")
                        raise
                else:
                    raise


# Recommended configurations for different scenarios
CONSERVATIVE_CONFIG = {
    'max_workers': 1,
    'base_delay': 5.0,
    'max_retries': 5,
    'min_request_interval': 0.5
}

MODERATE_CONFIG = {
    'max_workers': 2,
    'base_delay': 3.0,
    'max_retries': 3,
    'min_request_interval': 0.2
}

AGGRESSIVE_CONFIG = {
    'max_workers': 3,
    'base_delay': 2.0,
    'max_retries': 2,
    'min_request_interval': 0.1
}


def get_config(mode: str = 'moderate') -> dict:
    """
    Get rate limit configuration.
    
    Args:
        mode: 'conservative', 'moderate', or 'aggressive'
        
    Returns:
        Configuration dictionary
    """
    configs = {
        'conservative': CONSERVATIVE_CONFIG,
        'moderate': MODERATE_CONFIG,
        'aggressive': AGGRESSIVE_CONFIG
    }
    return configs.get(mode, MODERATE_CONFIG)


# Usage tips
USAGE_TIPS = """
Rate Limit Handling Tips:
========================

1. For OpenAI API with default tier:
   - Use --workers 1 or 2 to avoid rate limits
   - Add --no-judge flag to skip LLM judge evaluation (saves tokens)
   
2. If you keep hitting rate limits:
   python eval/infer_base_llm.py --workers 1 --no-judge
   
3. For faster processing with higher tier:
   python eval/infer_base_llm.py --workers 3
   
4. Monitor your usage at:
   https://platform.openai.com/account/rate-limits
"""

if __name__ == "__main__":
    print(USAGE_TIPS)