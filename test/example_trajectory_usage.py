"""
Example usage of the L-MARS trajectory tracking system.
Demonstrates how to use the enhanced workflow with result saving.
"""
import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from lmars.graph import create_legal_mind_graph
from test.results_viewer import ResultsViewer


def run_legal_research_with_tracking():
    """Example of running legal research with trajectory tracking enabled."""
    
    # Create L-MARS instance with trajectory tracking enabled (default)
    lmars = create_legal_mind_graph(llm_model="openai:gpt-4")
    
    # Example legal query
    user_query = "Can an F1 student work remotely for a US company while studying?"
    
    print("=== L-MARS Legal Research with Trajectory Tracking ===")
    print(f"Query: {user_query}")
    print("-" * 60)
    
    # Run the research and capture the trajectory
    try:
        # Using stream mode to see real-time progress
        config = {"configurable": {"thread_id": "demo_session"}}
        
        run_id = None
        for event in lmars.stream(user_query, config):
            current_step = event.get("current_step", "unknown")
            run_id = event.get("run_id")
            
            print(f"Step: {current_step}")
            
            if "messages" in event and event["messages"]:
                last_message = event["messages"][-1]
                if hasattr(last_message, 'content'):
                    content = last_message.content
                    # Truncate long content for display
                    if len(content) > 200:
                        content = content[:200] + "..."
                    print(f"Output: {content}")
            
            print("-" * 30)
        
        print(f"\\n✅ Research completed! Trajectory saved with ID: {run_id}")
        
        # Demonstrate trajectory viewing
        if run_id:
            print("\\n=== Trajectory Analysis ===")
            viewer = ResultsViewer()
            viewer.print_run_summary(run_id)
            
            # Export detailed trajectory
            export_file = viewer.export_run(run_id)
            print(f"\\n📄 Detailed trajectory exported to: {export_file}")
        
    except Exception as e:
        print(f"Error during research: {e}")


def demonstrate_trajectory_viewer():
    """Demonstrate the trajectory viewer functionality."""
    
    print("\\n=== Trajectory Viewer Demo ===")
    viewer = ResultsViewer()
    
    # List all available runs
    runs = viewer.list_runs()
    if not runs:
        print("No trajectory runs found. Run some legal research first!")
        return
    
    print(f"\\nFound {len(runs)} saved trajectories:")
    for i, run in enumerate(runs[:3], 1):  # Show first 3
        print(f"{i}. {run['run_id'][:8]}... - {run['start_time']} - {run['original_query'][:50]}...")
    
    # Show detailed view of the most recent run
    if runs:
        latest_run = runs[0]
        print(f"\\n=== Detailed View of Latest Run ===")
        viewer.print_run_summary(latest_run['run_id'])
        
        # Show model interactions
        interactions = viewer.view_model_interactions(latest_run['run_id'])
        if interactions:
            print(f"\\nModel Interactions ({len(interactions)} total):")
            for interaction in interactions[:2]:  # Show first 2
                print(f"- {interaction['step_name']}: {interaction['model_name']}")
                print(f"  Input: {interaction['input_prompt']}")
                print(f"  Output: {interaction['output_response']}")
                print()


def run_without_tracking():
    """Example of running without trajectory tracking."""
    
    print("\\n=== L-MARS without Trajectory Tracking ===")
    
    # Create L-MARS instance with tracking disabled
    lmars = create_legal_mind_graph(llm_model="openai:gpt-4")
    lmars.tracker = None  # Disable tracking
    
    user_query = "What are the employment restrictions for H1B visa holders?"
    
    print(f"Query: {user_query}")
    print("-" * 60)
    
    try:
        # Simple invoke without streaming
        result = lmars.invoke(user_query)
        
        print("Research completed (no trajectory saved)")
        if "final_answer" in result:
            final_answer = result["final_answer"]
            print(f"\\nAnswer: {final_answer.answer[:200]}...")
        
    except Exception as e:
        print(f"Error during research: {e}")


if __name__ == "__main__":
    # Run legal research with trajectory tracking
    run_legal_research_with_tracking()
    
    # Demonstrate trajectory viewer
    demonstrate_trajectory_viewer()
    
    # Show example without tracking
    run_without_tracking()
    
    print("\\n=== Usage Notes ===")
    print("1. Trajectories are saved in the 'results/' directory")
    print("2. Each run gets a unique UUID for identification")
    print("3. Use results_viewer.py for detailed trajectory analysis")
    print("4. Tracking can be disabled by setting enable_tracking=False")
    print("5. Use 'python results_viewer.py list' to see all saved runs")
    print("6. Use 'python results_viewer.py view <run_id>' for detailed analysis")