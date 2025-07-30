#!/usr/bin/env python3
"""
End-to-end CLI test runner for L-MARS legal research system.
Interactive command-line interface for testing the complete workflow.
"""
import sys
import os
from typing import Dict, Any, List

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from lmars.graph import create_legal_mind_graph
from test.results_viewer import ResultsViewer


class CLITestRunner:
    """Command-line interface for testing L-MARS workflow."""
    
    def __init__(self):
        self.lmars = None
        self.viewer = ResultsViewer()
        self.current_config = {"configurable": {"thread_id": "cli_test_session"}}
        
    def print_header(self):
        """Print application header."""
        print("=" * 80)
        print("🏛️  L-MARS: Legal Multi-Agent Research System - CLI Test Runner")
        print("=" * 80)
        print("Interactive command-line interface for end-to-end testing")
        print("Type 'help' for available commands or 'quit' to exit")
        print("-" * 80)
    
    def print_help(self):
        """Print available commands."""
        commands = {
            "query <question>": "Start a new legal research query",
            "continue": "Continue with follow-up questions if paused",
            "list": "List all saved trajectory runs",
            "view <run_id>": "View details of a specific run",
            "export <run_id>": "Export run to detailed JSON",
            "config": "Show current configuration",
            "model <model_name>": "Change the LLM model (e.g., 'model openai:gpt-4o')",
            "judge <model_name>": "Change the judge model (e.g., 'judge openai:o3-mini')",
            "clear": "Clear the screen",
            "help": "Show this help message",
            "quit": "Exit the application"
        }
        
        print("\n📋 Available Commands:")
        print("-" * 50)
        for cmd, desc in commands.items():
            print(f"  {cmd:<20} - {desc}")
        print()
    
    def initialize_system(self, model: str = "openai:gpt-4o"):
        """Initialize the L-MARS system."""
        try:
            print(f"🔧 Initializing L-MARS with model: {model}")
            # Use gpt-4o for both main and judge for compatibility
            self.lmars = create_legal_mind_graph(llm_model=model, judge_model=model)
            print("✅ System initialized successfully!")
            return True
        except Exception as e:
            print(f"❌ Failed to initialize system: {e}")
            return False
    
    def handle_query(self, query: str):
        """Handle a new legal research query."""
        if not self.lmars:
            print("❌ System not initialized. Please restart the application.")
            return
        
        print(f"\n🔍 Starting legal research...")
        print(f"Query: {query}")
        print("=" * 60)
        
        try:
            run_id = None
            step_count = 0
            
            # Stream the research process
            for event in self.lmars.stream(query, self.current_config):
                step_count += 1
                current_step = event.get("current_step", "unknown")
                run_id = event.get("run_id")
                
                print(f"\n📍 Step {step_count}: {current_step.replace('_', ' ').title()}")
                print("-" * 40)
                
                # Handle different types of outputs
                if "messages" in event and event["messages"]:
                    last_message = event["messages"][-1]
                    if hasattr(last_message, 'content'):
                        content = last_message.content
                        
                        # Check if this is follow-up questions (needs user input)
                        if "clarifying questions" in content.lower():
                            print("🤔 Follow-up questions generated:")
                            print(content)
                            print("\n⏸️  Workflow paused for user input.")
                            print("💡 Use 'continue' command to provide answers and resume.")
                            return
                        else:
                            print(f"📄 Output:")
                            print(content)  # Show full content, no truncation
                
                # Show search results count
                if "search_results" in event:
                    results_count = len(event["search_results"])
                    print(f"🔎 Found {results_count} search results")
                
                # Show judgment
                if "judgment" in event and event["judgment"]:
                    judgment = event["judgment"]
                    sufficient = "✅ Sufficient" if judgment.is_sufficient else "❌ Insufficient"
                    print(f"⚖️  Judge evaluation: {sufficient}")
                
                print("-" * 40)
            
            print(f"\n🎉 Research completed!")
            if run_id:
                print(f"📊 Trajectory saved with ID: {run_id[:8]}...")
                self._show_run_summary(run_id)
            
        except KeyboardInterrupt:
            print("\n⚠️  Research interrupted by user")
        except Exception as e:
            print(f"❌ Error during research: {e}")
    
    def handle_continue(self):
        """Handle continuation after follow-up questions."""
        if not self.lmars:
            print("❌ System not initialized.")
            return
        
        print("\n📝 Please provide answers to the follow-up questions:")
        print("Enter your responses (press Enter twice when done):")
        
        responses = {}
        question_num = 1
        
        while True:
            try:
                response = input(f"Answer {question_num}: ").strip()
                if not response:
                    break
                responses[f"question_{question_num}"] = response
                question_num += 1
            except KeyboardInterrupt:
                print("\n⚠️  Input cancelled")
                return
        
        if not responses:
            print("❌ No responses provided")
            return
        
        print(f"\n🔄 Continuing research with {len(responses)} responses...")
        print("=" * 60)
        
        try:
            step_count = 0
            for event in self.lmars.continue_conversation(responses, self.current_config):
                step_count += 1
                current_step = event.get("current_step", "unknown")
                
                print(f"\n📍 Step {step_count}: {current_step.replace('_', ' ').title()}")
                print("-" * 40)
                
                if "messages" in event and event["messages"]:
                    last_message = event["messages"][-1]
                    if hasattr(last_message, 'content'):
                        print(f"📄 Output:")
                        print(last_message.content)  # Show full content, no truncation
                
                if "search_results" in event:
                    results_count = len(event["search_results"])
                    print(f"🔎 Found {results_count} additional search results")
                
                if "judgment" in event and event["judgment"]:
                    judgment = event["judgment"]
                    sufficient = "✅ Sufficient" if judgment.is_sufficient else "❌ Insufficient"
                    print(f"⚖️  Judge evaluation: {sufficient}")
                
                print("-" * 40)
            
            print(f"\n🎉 Research completed after follow-up!")
            
        except Exception as e:
            print(f"❌ Error during continuation: {e}")
    
    def handle_list_runs(self):
        """List all saved trajectory runs."""
        runs = self.viewer.list_runs()
        
        if not runs:
            print("📂 No saved trajectory runs found")
            return
        
        print(f"\n📊 Found {len(runs)} saved trajectory runs:")
        print("=" * 80)
        print(f"{'ID':<10} {'Date/Time':<20} {'Query':<45}")
        print("-" * 80)
        
        for run in runs[:10]:  # Show latest 10
            run_id_short = run['run_id'][:8] + "..."
            query_short = run['original_query'][:42] + "..." if len(run['original_query']) > 42 else run['original_query']
            start_time = run['start_time'][:19].replace('T', ' ')  # Remove microseconds
            print(f"{run_id_short:<10} {start_time:<20} {query_short:<45}")
        
        if len(runs) > 10:
            print(f"... and {len(runs) - 10} more runs")
        print()
    
    def handle_view_run(self, run_id: str):
        """View details of a specific run."""
        # Allow partial run_id matching
        runs = self.viewer.list_runs()
        matching_run = None
        
        for run in runs:
            if run['run_id'].startswith(run_id) or run_id in run['run_id']:
                matching_run = run['run_id']
                break
        
        if not matching_run:
            print(f"❌ No run found matching '{run_id}'")
            return
        
        self._show_run_summary(matching_run)
    
    def handle_export_run(self, run_id: str):
        """Export a run to detailed JSON."""
        try:
            # Allow partial run_id matching
            runs = self.viewer.list_runs()
            matching_run = None
            
            for run in runs:
                if run['run_id'].startswith(run_id) or run_id in run['run_id']:
                    matching_run = run['run_id']
                    break
            
            if not matching_run:
                print(f"❌ No run found matching '{run_id}'")
                return
            
            export_file = self.viewer.export_run(matching_run)
            print(f"📄 Exported run to: {export_file}")
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
    
    def _show_run_summary(self, run_id: str):
        """Show a summary of a trajectory run."""
        run_info = self.viewer.view_run(run_id)
        if not run_info:
            print(f"❌ Run {run_id} not found")
            return
        
        print(f"\n📊 Trajectory Summary")
        print("=" * 50)
        print(f"🆔 Run ID: {run_id[:8]}...")
        print(f"⏰ Duration: {run_info['duration']}")
        print(f"📝 Query: {run_info['original_query']}")
        print(f"🔄 Steps: {run_info['steps_count']}")
        print(f"👤 Human Inputs: {run_info['human_inputs_count']}")
        
        # Show workflow steps
        steps = self.viewer.view_run_steps(run_id)
        if steps:
            print(f"\n📋 Workflow Steps:")
            for i, step in enumerate(steps, 1):
                print(f"  {i}. {step['node_name'].replace('_', ' ').title()}")
                print(f"     ⏱️  {step['duration_seconds']:.2f}s, 🤖 {step['model_calls_count']} model calls")
        
        print("-" * 50)
    
    def handle_config(self):
        """Show current configuration."""
        print("\n⚙️  Current Configuration:")
        print("-" * 30)
        if self.lmars:
            print(f"🤖 Model: {self.lmars.llm}")
            print(f"🔄 Max Iterations: {self.lmars.max_iterations}")
            print(f"📊 Tracking: {'Enabled' if self.lmars.tracker else 'Disabled'}")
        else:
            print("❌ System not initialized")
        print(f"🔗 Session ID: {self.current_config['configurable']['thread_id']}")
        print()
    
    def change_model(self, model_name: str):
        """Change the LLM model."""
        print(f"🔄 Changing model to: {model_name}")
        if self.initialize_system(model_name):
            print("✅ Model changed successfully!")
        else:
            print("❌ Failed to change model")
    
    def change_judge_model(self, judge_model: str):
        """Change the judge model specifically."""
        print(f"🔄 Changing judge model to: {judge_model}")
        try:
            # Get current main model
            current_model = "openai:gpt-4o"  # Default fallback
            if self.lmars:
                current_model = str(self.lmars.llm)
            
            # Reinitialize with new judge model
            self.lmars = create_legal_mind_graph(llm_model=current_model, judge_model=judge_model)
            print("✅ Judge model changed successfully!")
        except Exception as e:
            print(f"❌ Failed to change judge model: {e}")
            print("💡 Note: o3-mini may have compatibility issues, try using gpt-4o instead")
    
    def run(self):
        """Main CLI loop."""
        self.print_header()
        
        # Initialize system
        if not self.initialize_system():
            print("❌ Failed to initialize. Exiting.")
            return
        
        # Main command loop
        while True:
            try:
                user_input = input("\n🏛️  L-MARS> ").strip()
                
                if not user_input:
                    continue
                
                # Parse command
                parts = user_input.split(None, 1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""
                
                # Handle commands
                if command == "quit" or command == "exit":
                    print("👋 Goodbye!")
                    break
                
                elif command == "help":
                    self.print_help()
                
                elif command == "clear":
                    os.system('clear' if os.name == 'posix' else 'cls')
                    self.print_header()
                
                elif command == "query":
                    if not args:
                        print("❌ Please provide a legal question. Usage: query <your question>")
                    else:
                        self.handle_query(args)
                
                elif command == "continue":
                    self.handle_continue()
                
                elif command == "list":
                    self.handle_list_runs()
                
                elif command == "view":
                    if not args:
                        print("❌ Please provide a run ID. Usage: view <run_id>")
                    else:
                        self.handle_view_run(args)
                
                elif command == "export":
                    if not args:
                        print("❌ Please provide a run ID. Usage: export <run_id>")
                    else:
                        self.handle_export_run(args)
                
                elif command == "config":
                    self.handle_config()
                
                elif command == "model":
                    if not args:
                        print("❌ Please provide a model name. Usage: model <model_name>")
                    else:
                        self.change_model(args)
                
                elif command == "judge":
                    if not args:
                        print("❌ Please provide a judge model name. Usage: judge <model_name>")
                    else:
                        self.change_judge_model(args)
                
                else:
                    print(f"❌ Unknown command: {command}")
                    print("💡 Type 'help' for available commands")
            
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except EOFError:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")


def main():
    """Entry point for the CLI test runner."""
    # Check if we have required environment variables or API keys
    print("🔍 Checking system requirements...")
    from dotenv import load_dotenv
    load_dotenv()
    # Basic environment check
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  Warning: No API keys found in environment variables")
        print("   Make sure to set OPENAI_API_KEY or ANTHROPIC_API_KEY")
        print("   The system may not work without proper API credentials")
        
        response = input("\n❓ Continue anyway? (y/N): ").strip().lower()
        if response != 'y':
            print("👋 Exiting...")
            return
    
    # Create and run the CLI
    cli = CLITestRunner()
    cli.run()


if __name__ == "__main__":
    main()