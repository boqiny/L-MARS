"""
Configuration settings for L-MARS Streamlit application.
"""
import os
from typing import Dict, List


class AppConfig:
    """Application configuration constants."""
    
    # App metadata
    APP_NAME = "L-MARS Legal Research System"
    APP_VERSION = "1.0.0"
    APP_DESCRIPTION = "Multi-Agent Legal Research with Advanced Reasoning"
    
    # Page configuration
    PAGE_TITLE = "L-MARS Legal Research"
    PAGE_ICON = "🏛️"
    LAYOUT = "wide"
    
    # Model options
    MAIN_MODEL_OPTIONS = [
        "openai:gpt-4o",
        "openai:gpt-4",
        "openai:gpt-4-turbo",
        "openai:gpt-3.5-turbo",
        "anthropic:claude-3-opus-20240229",
        "anthropic:claude-3-sonnet-20240229"
    ]
    
    JUDGE_MODEL_OPTIONS = [
        "openai:gpt-4o",
        "openai:gpt-4",
        "openai:o3-mini",
        "anthropic:claude-3-opus-20240229"
    ]
    
    # Default settings
    DEFAULT_MAIN_MODEL = "openai:gpt-4o"
    DEFAULT_JUDGE_MODEL = "openai:gpt-4o"
    DEFAULT_MAX_ITERATIONS = 3
    
    # UI settings
    MAX_QUERY_LENGTH = 2000
    MAX_FOLLOWUP_RESPONSES = 10
    TRAJECTORY_DISPLAY_LIMIT = 10
    
    # Example queries for user guidance
    EXAMPLE_QUERIES = [
        "Can an F1 student work remotely for a US company while studying?",
        "What are the legal requirements for starting an LLC in California?",
        "Is it legal to record a conversation without consent in New York?",
        "What are the penalties for copyright infringement for commercial use?",
        "Can I be sued for negative online reviews of a business?",
        "What constitutes fair use in trademark law?",
        "Are non-compete agreements enforceable after termination?",
        "What are the legal obligations of landlords for rental property maintenance?"
    ]
    
    # Error messages
    ERROR_MESSAGES = {
        "no_api_key": "Please set your API keys in environment variables (OPENAI_API_KEY, ANTHROPIC_API_KEY)",
        "system_init_failed": "Failed to initialize the L-MARS system. Please check your configuration.",
        "query_too_long": f"Query is too long. Please limit to {MAX_QUERY_LENGTH} characters.",
        "empty_query": "Please enter a legal question to get started.",
        "research_failed": "Research failed. Please try again or contact support.",
        "invalid_model": "Selected model is not available. Please choose a different model."
    }
    
    # Success messages
    SUCCESS_MESSAGES = {
        "system_initialized": "L-MARS system initialized successfully!",
        "research_completed": "Legal research completed successfully!",
        "settings_saved": "Configuration settings saved successfully!",
        "export_completed": "Data exported successfully!"
    }
    
    # Styling
    CUSTOM_CSS = """
    <style>
    .main-header {
        padding: 1rem 0;
        border-bottom: 2px solid #f0f2f6;
        margin-bottom: 2rem;
    }
    
    .status-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .status-ready { background: #d4edda; color: #155724; }
    .status-running { background: #fff3cd; color: #856404; }
    .status-paused { background: #f8d7da; color: #721c24; }
    .status-completed { background: #d1ecf1; color: #0c5460; }
    
    .step-container {
        border: 1px solid #e9ecef;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        background: #f8f9fa;
    }
    
    .confidence-high { color: #28a745; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #dc3545; }
    
    .sidebar .sidebar-content {
        padding-top: 1rem;
    }
    
    .legal-disclaimer {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
    """
    
    @classmethod
    def get_model_display_name(cls, model_key: str) -> str:
        """Get a user-friendly display name for a model."""
        display_names = {
            "openai:gpt-4o": "GPT-4o (Latest)",
            "openai:gpt-4": "GPT-4 (Standard)",
            "openai:gpt-4-turbo": "GPT-4 Turbo",
            "openai:gpt-3.5-turbo": "GPT-3.5 Turbo",
            "openai:o3-mini": "O3-Mini (Experimental)",
            "anthropic:claude-3-opus-20240229": "Claude 3 Opus",
            "anthropic:claude-3-sonnet-20240229": "Claude 3 Sonnet"
        }
        return display_names.get(model_key, model_key)
    
    @classmethod
    def validate_environment(cls) -> Dict[str, bool]:
        """Validate required environment variables."""
        checks = {
            "openai_key": bool(os.getenv("OPENAI_API_KEY")),
            "anthropic_key": bool(os.getenv("ANTHROPIC_API_KEY")),
            "has_any_key": bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))
        }
        return checks
    
    @classmethod
    def get_available_models(cls) -> Dict[str, List[str]]:
        """Get available models based on environment setup."""
        env_checks = cls.validate_environment()
        
        available = {
            "main": [],
            "judge": []
        }
        
        # Add OpenAI models if key is available
        if env_checks["openai_key"]:
            openai_models = [m for m in cls.MAIN_MODEL_OPTIONS if m.startswith("openai:")]
            available["main"].extend(openai_models)
            
            openai_judge_models = [m for m in cls.JUDGE_MODEL_OPTIONS if m.startswith("openai:")]
            available["judge"].extend(openai_judge_models)
        
        # Add Anthropic models if key is available
        if env_checks["anthropic_key"]:
            anthropic_models = [m for m in cls.MAIN_MODEL_OPTIONS if m.startswith("anthropic:")]
            available["main"].extend(anthropic_models)
            
            anthropic_judge_models = [m for m in cls.JUDGE_MODEL_OPTIONS if m.startswith("anthropic:")]
            available["judge"].extend(anthropic_judge_models)
        
        # Fallback to defaults if no keys are available (for testing)
        if not available["main"]:
            available["main"] = cls.MAIN_MODEL_OPTIONS
            available["judge"] = cls.JUDGE_MODEL_OPTIONS
        
        return available