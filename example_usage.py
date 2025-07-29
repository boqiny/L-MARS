"""
Example usage of the multi-agent legal research system.
"""
import os
from lmars.graph import create_legal_mind_graph

def interactive_legal_research():
    """Interactive example of using the legal research system."""
    
    print("🏛️  L-MARS Multi-Agent Research System")
    print("=" * 50)
    
    # Check for required environment variables
    required_vars = ["OPENAI_API_KEY", "SERPER_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("\nPlease set these environment variables:")
        for var in missing_vars:
            print(f"export {var}='your_api_key_here'")
        return
    
    try:
        # Initialize the system
        print("🔧 Initializing agents...")
        lmars = create_legal_mind_graph("openai:gpt-4")
        print("✅ System ready!")
        
        while True:
            print("\n" + "-" * 50)
            user_query = input("\n💬 Enter your legal question (or 'quit' to exit): ")
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_query.strip():
                print("Please enter a valid question.")
                continue
            
            print(f"\n🔍 Researching: {user_query}")
            print("=" * 60)
            
            # Create a unique session for this query
            import time
            session_id = f"session_{int(time.time())}"
            config = {"configurable": {"thread_id": session_id}}
            
            # Process the query
            try:
                for event in lmars.stream(user_query, config):
                    step = event.get("current_step", "")
                    
                    if step == "follow_up_questions":
                        # Check if we have follow-up questions
                        questions = event.get("follow_up_questions", [])
                        if questions and "messages" in event:
                            print("\n🤔 " + event["messages"][-1].content)
                            
                            # Collect user responses
                            responses = {}
                            for i, q in enumerate(questions, 1):
                                response = input(f"\nAnswer to question {i}: ")
                                responses[f"question_{i}"] = response
                            
                            # Continue with user responses
                            print("\n🔄 Continuing research with your responses...")
                            final_result = lmars.continue_conversation(responses, config)
                            
                            # Print final answer
                            if "messages" in final_result:
                                final_message = final_result["messages"][-1]
                                if hasattr(final_message, 'content'):
                                    print("\n" + "=" * 60)
                                    print("📋 FINAL RESEARCH RESULTS")
                                    print("=" * 60)
                                    print(final_message.content)
                    
                    elif step == "complete":
                        # Direct completion without follow-up questions
                        if "messages" in event:
                            final_message = event["messages"][-1]
                            if hasattr(final_message, 'content'):
                                print("\n" + "=" * 60)
                                print("📋 RESEARCH RESULTS")
                                print("=" * 60)
                                print(final_message.content)
            
            except Exception as e:
                print(f"❌ Error during research: {e}")
                continue
    
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")

def demo_example():
    """Run a demo with a predefined example."""
    
    print("🎯 Running Demo Example")
    print("=" * 30)
    
    # Example query
    demo_query = "Can an F1 student work part-time for a startup while studying computer science?"
    
    print(f"Demo Query: {demo_query}")
    print("\n🔄 Processing...")
    
    try:
        lmars = create_legal_mind_graph("openai:gpt-4")
        config = {"configurable": {"thread_id": "demo"}}
        
        # Stream the results
        for event in lmars.stream(demo_query, config):
            step = event.get("current_step", "")
            
            if step == "follow_up_questions":
                print("\n🤔 System generated follow-up questions:")
                questions = event.get("follow_up_questions", [])
                for i, q in enumerate(questions, 1):
                    print(f"   {i}. {q.question}")
                    print(f"      Reason: {q.reason}")
                
                # Auto-answer for demo
                demo_responses = {
                    "question_1": "California, USA",
                    "question_2": "Computer Science Masters program", 
                    "question_3": "Part-time, 10-15 hours per week"
                }
                
                print(f"\n🤖 Demo responses: {demo_responses}")
                print("\n🔄 Continuing with demo responses...")
                
                result = lmars.continue_conversation(demo_responses, config)
                
                if "messages" in result:
                    final_msg = result["messages"][-1]
                    if hasattr(final_msg, 'content'):
                        print("\n" + "=" * 60)
                        print("📋 DEMO RESULTS")
                        print("=" * 60)
                        print(final_msg.content)
                break
    
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    print("Choose an option:")
    print("1. Interactive Mode")
    print("2. Demo Example")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        interactive_legal_research()
    elif choice == "2":
        demo_example()
    else:
        print("Invalid choice. Running interactive mode by default.")
        interactive_legal_research()