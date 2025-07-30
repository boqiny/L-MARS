#!/usr/bin/env python3
"""
Test script for QueryAgent in both plain text and structured output modes.
"""

import os
import sys
import json
from dotenv import load_dotenv

# Add parent directory to path to import lmars
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmars.agents import QueryAgent, QueryResult
from langchain.chat_models import init_chat_model


# Test case: Emma's online shopping scam
TEST_CASE = """Emma was scrolling through social media when she saw a flashy online promotion: "Limited-Time Mega Sale – Up to 80% Off!" The website looked legitimate, filled with popular brand logos, countdown timers, and customer reviews. Without thinking too much, she placed an order for several items—clothes, accessories, and a gadget—spending over $200.

But after the purchase, she received no confirmation email. The customer service link was broken, and the tracking number never worked. That's when she realized she had been scammed. The website disappeared a few days later, and her attempts to get a refund through her bank failed because the transaction was processed as a debit and not covered by fraud protection.

Feeling anxious and frustrated, Emma didn't know what to do. She felt tricked and blamed herself for being careless."""


def setup_llm():
    """Set up the language model for testing."""
    llm = init_chat_model("gpt-4o", model_provider="openai")
    print("✅ Using OpenAI GPT-4o")
    return llm


def test_plain_text_mode(llm):
    """Test QueryAgent in plain text mode (default)."""
    print("Testing QueryAgent in Plain Text Mode")
    print("=" * 50)
    
    # Create agent with default plain text output
    agent = QueryAgent(llm)  # structured_output=False by default
    
    try:
        result = agent.analyze_query(TEST_CASE)
        
        print("📝 Plain Text Analysis Result:")
        print("-" * 30)
        print(result)
        print("-" * 30)
        
        # Verify it's a string
        assert isinstance(result, str), f"Expected string, got {type(result)}"
        print("✅ Plain text mode test passed")
        
        return result
        
    except Exception as e:
        print(f"❌ Plain text mode test failed: {e}")
        return None


def test_structured_mode(llm):
    """Test QueryAgent in structured output mode."""
    print("Testing QueryAgent in Structured Output Mode")
    print("=" * 50)
    
    # Create agent with structured output enabled
    agent = QueryAgent(llm, structured_output=True)
    
    try:
        result = agent.analyze_query(TEST_CASE)
        
        print("📊 Structured Analysis Result:")
        print("-" * 30)
        
        # Verify it's a QueryResult
        assert isinstance(result, QueryResult), f"Expected QueryResult, got {type(result)}"
        try:
            json_output = result.model_dump()
            print(json.dumps(json_output, indent=2, ensure_ascii=False))
        except Exception as json_error:
            print(f"❌ Failed to convert to JSON: {json_error}")
        
        print("-" * 30)
        print("✅ Structured mode test passed")
        
        return result
        
    except Exception as e:
        print(f"❌ Structured mode test failed: {e}")
        return None


def main():
    """Main test function."""
    print("🧪 QueryAgent Test Suite")
    print("=" * 60)
    
    print("📋 Default agent mode: Plain Text (structured_output=False)")
    load_dotenv()
    # Set up LLM
    llm = setup_llm()
    
    # Test both modes
    plain_result = test_plain_text_mode(llm)
    structured_result = test_structured_mode(llm)
    
    print(f"\n🎯 Summary")
    print("=" * 50)
    print(f"✅ Plain text mode (default): {'Passed' if plain_result else 'Failed'}")
    print(f"✅ Structured mode (opt-in): {'Passed' if structured_result else 'Failed'}")
    print("📊 Agent defaults to plain text output unless structured_output=True")


if __name__ == "__main__":
    main()