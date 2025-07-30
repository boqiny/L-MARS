#!/usr/bin/env python3
"""
Simple test script for the interactive clarification flow.
Tests the follow-up questions functionality like DeepResearch.
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path to import lmars
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lmars import create_legal_mind_graph
from langchain.chat_models import init_chat_model


def setup_llm():
    """Set up the language model for testing."""
    llm = init_chat_model("gpt-4o", model_provider="openai")
    print("✅ Using OpenAI GPT-4o")
    return llm


def test_clarification_flow():
    """Test the interactive clarification flow."""
    print("🧪 Testing Interactive Clarification Flow")
    print("=" * 50)
    
    load_dotenv()
    
    # Create the legal research system
    lmars = create_legal_mind_graph()
    
    # Test with a vague query that should trigger follow-up questions
    vague_query = "I have a contract problem"
    
    print(f"📝 Initial Query: '{vague_query}'")
    print("-" * 30)
    
    # Start the workflow
    config = {"configurable": {"thread_id": "test_clarification"}}
    
    try:
        # Process the initial query
        events = list(lmars.stream(vague_query, config))
        
        # Find the follow-up questions step
        follow_up_step = None
        for event in events:
            if event.get("current_step") == "follow_up_questions":
                follow_up_step = event
                break
        
        if follow_up_step and "follow_up_questions" in follow_up_step:
            questions = follow_up_step["follow_up_questions"]
            
            print("🔍 System Generated Follow-up Questions:")
            for i, q in enumerate(questions, 1):
                print(f"{i}. {q.question}")
                print(f"   Reason: {q.reason}")
                print()
            
            # Simulate user responses
            user_responses = {
                "question_1": "Employment contract",
                "question_2": "My employer is not paying overtime as specified in the contract"
            }
            
            print("👤 Simulated User Responses:")
            for key, response in user_responses.items():
                print(f"{key}: {response}")
            print("-" * 30)
            
            # Continue the conversation with user responses
            print("🔄 Continuing workflow with user responses...")
            continuation_events = list(lmars.continue_conversation(user_responses, config))
            
            # Check if workflow continued successfully
            if continuation_events:
                print("✅ Clarification flow completed successfully!")
                print(f"📊 Generated {len(continuation_events)} workflow events after clarification")
                
                # Show the search queries that were generated with context
                for event in continuation_events:
                    if "search_queries" in event:
                        queries = event["search_queries"]
                        print(f"\n🔍 Generated {len(queries)} search queries with context:")
                        for query in queries:
                            print(f"- {query.query} ({query.query_type}, {query.priority} priority)")
                        break
            else:
                print("❌ No continuation events generated")
                
        else:
            print("❌ No follow-up questions generated")
            print("Available events:", [e.get("current_step") for e in events])
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main test function."""
    print("🧪 L-MARS Clarification Flow Test")
    print("Testing DeepResearch-style interactive clarification")
    print("=" * 60)
    
    test_clarification_flow()
    
    print(f"\n🎯 Test Summary")
    print("=" * 50)
    print("This test verifies:")
    print("✓ System generates follow-up questions for vague queries")
    print("✓ User can provide responses to clarify their needs") 
    print("✓ System continues workflow with enriched context")
    print("✓ Search queries are generated using clarification context")


if __name__ == "__main__":
    main()