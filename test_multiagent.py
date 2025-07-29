"""
Test script for the multi-agent legal research system.
"""
import os
from lmars.graph import create_legal_mind_graph

def test_basic_workflow():
    """Test the basic workflow of the multi-agent system."""
    
    # Set up environment (you'll need to add your API keys)
    required_env_vars = ["OPENAI_API_KEY", "SERPER_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Missing environment variables: {missing_vars}")
        print("Please set these before running the test.")
        return
    
    try:
        # Create the legal research system
        print("Initializing Legal Research System...")
        legal_mind = create_legal_mind_graph("openai:gpt-4")
        
        # Test query
        test_query = "Can an F1 student work remotely for a US company while studying?"
        print(f"\nTest Query: {test_query}")
        print("=" * 60)
        
        # Run the workflow
        config = {"configurable": {"thread_id": "test_session"}}
        
        print("Streaming workflow steps:")
        print("-" * 40)
        
        for event in legal_mind.stream(test_query, config):
            step = event.get("current_step", "unknown")
            print(f"\n🔄 Step: {step}")
            
            # Print relevant information for each step
            if step == "query_processing":
                print(f"   Processing query: {event.get('original_query', '')[:50]}...")
            
            elif step == "follow_up_questions":
                questions = event.get("follow_up_questions", [])
                print(f"   Generated {len(questions)} follow-up questions")
                if "messages" in event and event["messages"]:
                    print(f"   Preview: {event['messages'][-1].content[:100]}...")
            
            elif step == "generate_queries":
                queries = event.get("search_queries", [])
                print(f"   Generated {len(queries)} search queries")
                for q in queries:
                    print(f"     - {q.query_type}: {q.query[:50]}...")
            
            elif step == "search_execution":
                results = event.get("search_results", [])
                print(f"   Found {len(results)} search results")
                for r in results:
                    print(f"     - {r.source}: {r.title[:40]}...")
            
            elif step == "judge_evaluation":
                judgment = event.get("judgment")
                if judgment:
                    print(f"   Sufficient: {judgment.is_sufficient}")
                    print(f"   Confidence: {judgment.confidence:.2f}")
            
            elif step == "complete":
                final_answer = event.get("final_answer")
                if final_answer:
                    print(f"   Generated final answer (confidence: {final_answer.confidence:.2f})")
                    print(f"   Key points: {len(final_answer.key_points)}")
                    print(f"   Sources: {len(final_answer.sources)}")
        
        print("\n" + "=" * 60)
        print("✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

def test_structured_output():
    """Test that structured output models work correctly."""
    
    from lmars.agents import FollowUpQuestion, QueryGeneration, FinalAnswer
    
    print("Testing structured output models...")
    
    # Test FollowUpQuestion
    question = FollowUpQuestion(
        question="What state are you studying in?",
        reason="Laws vary by state jurisdiction"
    )
    print(f"✅ FollowUpQuestion: {question.question}")
    
    # Test QueryGeneration
    query = QueryGeneration(
        query="F1 student remote work regulations",
        query_type="web_search",
        priority="high"
    )
    print(f"✅ QueryGeneration: {query.query} ({query.query_type})")
    
    # Test FinalAnswer
    answer = FinalAnswer(
        answer="Test answer",
        key_points=["Point 1", "Point 2"],
        sources=["Source 1"],
        confidence=0.8,
        disclaimers=["This is not legal advice"]
    )
    print(f"✅ FinalAnswer: {answer.answer} (confidence: {answer.confidence})")
    
    print("✅ All structured output models working correctly!")

if __name__ == "__main__":
    print("🧪 Testing Multi-Agent Legal Research System")
    print("=" * 50)
    
    # First test structured output
    test_structured_output()
    
    print("\n" + "=" * 50)
    
    # Then test the workflow
    test_basic_workflow()