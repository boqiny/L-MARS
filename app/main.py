"""
L-MARS Legal Research System - Streamlit Frontend
Interactive web interface for legal question answering with multi-agent workflow.
"""
import streamlit as st
import sys
import os
import time
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from lmars.graph import create_legal_mind_graph
from test.results_viewer import ResultsViewer


class LMarsStreamlitApp:
    """Streamlit application for L-MARS legal research system."""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
    
    def setup_page_config(self):
        """Configure Streamlit page settings."""
        st.set_page_config(
            page_title="L-MARS Legal Research",
            page_icon="🏛️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'lmars_system' not in st.session_state:
            st.session_state.lmars_system = None
        
        if 'current_query' not in st.session_state:
            st.session_state.current_query = ""
        
        if 'workflow_state' not in st.session_state:
            st.session_state.workflow_state = "idle"  # idle, running, paused, completed
        
        if 'follow_up_questions' not in st.session_state:
            st.session_state.follow_up_questions = []
        
        if 'workflow_steps' not in st.session_state:
            st.session_state.workflow_steps = []
        
        if 'current_run_id' not in st.session_state:
            st.session_state.current_run_id = None
        
        if 'final_answer' not in st.session_state:
            st.session_state.final_answer = None
        
        if 'config' not in st.session_state:
            st.session_state.config = {
                'llm_model': 'openai:gpt-4o',
                'judge_model': 'openai:gpt-4o',
                'max_iterations': 3
            }
    
    def render_header(self):
        """Render the application header."""
        st.title("🏛️ L-MARS Legal Research System")
        st.markdown("*Multi-Agent Legal Research with Advanced Reasoning*")
        
        # Status indicator
        status_colors = {
            "idle": "🟢 Ready",
            "running": "🟡 Processing...",
            "paused": "🟠 Awaiting Input",
            "completed": "✅ Complete"
        }
        
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            st.markdown(f"**Status:** {status_colors.get(st.session_state.workflow_state, '⚪ Unknown')}")
        
        with col2:
            if st.session_state.current_run_id:
                st.markdown(f"**Run ID:** `{st.session_state.current_run_id[:8]}...`")
        
        with col3:
            if st.button("🔄 Reset Session"):
                self.reset_session()
                st.rerun()
    
    def render_sidebar(self):
        """Render the sidebar with configuration and controls."""
        st.sidebar.header("⚙️ Configuration")
        
        # Model Selection
        st.sidebar.subheader("Model Settings")
        
        llm_model = st.sidebar.selectbox(
            "Main LLM Model",
            ["openai:gpt-4o", "openai:gpt-4", "openai:gpt-3.5-turbo"],
            index=0 if st.session_state.config['llm_model'] == 'openai:gpt-4o' else 1
        )
        
        judge_model = st.sidebar.selectbox(
            "Judge Model",
            ["openai:gpt-4o", "openai:gpt-4", "openai:o3-mini"],
            index=0 if st.session_state.config['judge_model'] == 'openai:gpt-4o' else 1
        )
        
        max_iterations = st.sidebar.slider(
            "Max Iterations",
            min_value=1,
            max_value=5,
            value=st.session_state.config['max_iterations']
        )
        
        # Update config if changed
        if (llm_model != st.session_state.config['llm_model'] or 
            judge_model != st.session_state.config['judge_model'] or
            max_iterations != st.session_state.config['max_iterations']):
            
            st.session_state.config.update({
                'llm_model': llm_model,
                'judge_model': judge_model,
                'max_iterations': max_iterations
            })
            
            # Reinitialize system
            st.session_state.lmars_system = None
        
        # API Key Status
        st.sidebar.subheader("🔑 API Keys")
        openai_key = bool(os.getenv("OPENAI_API_KEY"))
        anthropic_key = bool(os.getenv("ANTHROPIC_API_KEY"))
        
        if openai_key:
            st.sidebar.success("✅ OpenAI Key")
        else:
            st.sidebar.error("❌ OpenAI Key")
            
        if anthropic_key:
            st.sidebar.success("✅ Anthropic Key")
        else:
            st.sidebar.error("❌ Anthropic Key")
        
        if not openai_key and not anthropic_key:
            st.sidebar.warning("⚠️ No API keys found! Create a .env file.")
        
        # System Status
        st.sidebar.subheader("System Status")
        if st.session_state.lmars_system:
            st.sidebar.success("✅ System Initialized")
        else:
            st.sidebar.warning("⚠️ System Not Initialized")
            if st.sidebar.button("Initialize System"):
                self.initialize_system()
                st.rerun()
        
        # Trajectory Viewer
        st.sidebar.subheader("📊 Trajectory Analysis")
        if st.sidebar.button("View Past Runs"):
            st.session_state.show_trajectory_viewer = True
            st.rerun()
    
    def initialize_system(self):
        """Initialize the L-MARS system with current configuration."""
        try:
            # Check for API keys first
            openai_key = os.getenv("OPENAI_API_KEY")
            anthropic_key = os.getenv("ANTHROPIC_API_KEY")
            
            if not openai_key and not anthropic_key:
                st.error("🔑 No API keys found! Please set OPENAI_API_KEY or ANTHROPIC_API_KEY in your environment variables or .env file.")
                st.info("💡 Create a .env file in your project root with:\n```\nOPENAI_API_KEY=your_openai_key_here\n```")
                return
            
            # Check if selected models are compatible with available keys
            llm_model = st.session_state.config['llm_model']
            judge_model = st.session_state.config['judge_model']
            
            if llm_model.startswith("openai:") and not openai_key:
                st.error(f"🔑 OpenAI API key required for model: {llm_model}")
                return
                
            if judge_model.startswith("openai:") and not openai_key:
                st.error(f"🔑 OpenAI API key required for judge model: {judge_model}")
                return
                
            if llm_model.startswith("anthropic:") and not anthropic_key:
                st.error(f"🔑 Anthropic API key required for model: {llm_model}")
                return
                
            if judge_model.startswith("anthropic:") and not anthropic_key:
                st.error(f"🔑 Anthropic API key required for judge model: {judge_model}")
                return
            
            with st.spinner("Initializing L-MARS system..."):
                st.session_state.lmars_system = create_legal_mind_graph(
                    llm_model=st.session_state.config['llm_model'],
                    judge_model=st.session_state.config['judge_model'],
                    max_iterations=st.session_state.config['max_iterations']
                )
            st.success("✅ System initialized successfully!")
            
        except Exception as e:
            error_msg = str(e)
            if "api_key" in error_msg.lower():
                st.error("🔑 API Key Error: Please check your API keys in the .env file")
                st.info("💡 Make sure your .env file contains valid API keys:\n```\nOPENAI_API_KEY=sk-...\nANTHROPIC_API_KEY=sk-ant-...\n```")
            else:
                st.error(f"❌ Failed to initialize system: {e}")
                st.info("💡 Try refreshing the page or check the system logs")
    
    def render_query_input(self):
        """Render the main query input interface."""
        st.header("💬 Ask Your Legal Question")
        
        # Query input
        query = st.text_area(
            "Enter your legal question:",
            value=st.session_state.current_query,
            height=100,
            placeholder="e.g., Can an F1 student work remotely for a US company while studying?",
            disabled=st.session_state.workflow_state == "running"
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button(
                "🚀 Start Research", 
                disabled=(not query.strip() or 
                         st.session_state.workflow_state == "running" or
                         not st.session_state.lmars_system),
                type="primary"
            ):
                self.start_research(query)
        
        with col2:
            if st.session_state.workflow_state == "running":
                st.info("🔄 Research in progress... Please wait.")
            elif not st.session_state.lmars_system:
                st.warning("⚠️ Please initialize the system first (see sidebar)")
    
    def render_follow_up_questions(self):
        """Render follow-up questions interface."""
        if st.session_state.workflow_state == "paused" and st.session_state.follow_up_questions:
            st.header("❓ Follow-up Questions")
            st.info("The system needs more information to provide better assistance:")
            
            responses = {}
            
            for i, question in enumerate(st.session_state.follow_up_questions, 1):
                st.markdown(f"**Question {i}:** {question.question}")
                st.markdown(f"*{question.reason}*")
                
                response = st.text_input(
                    f"Your answer to question {i}:",
                    key=f"followup_{i}",
                    placeholder="Enter your response..."
                )
                
                if response.strip():
                    responses[f"question_{i}"] = response.strip()
                
                st.markdown("---")
            
            col1, col2 = st.columns([1, 4])
            
            with col1:
                if st.button(
                    "📤 Submit Answers", 
                    disabled=len(responses) == 0,
                    type="primary"
                ):
                    self.continue_research(responses)
            
            with col2:
                if len(responses) == 0:
                    st.warning("⚠️ Please provide at least one answer to continue")
    
    def render_workflow_progress(self):
        """Render real-time workflow progress."""
        if st.session_state.workflow_steps:
            st.header("⚡ Workflow Progress")
            
            progress_container = st.container()
            
            with progress_container:
                for i, step in enumerate(st.session_state.workflow_steps):
                    step_name = step.get('step_name', 'Unknown Step')
                    status = step.get('status', 'pending')
                    output = step.get('output', '')
                    
                    # Step header
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        if status == 'completed':
                            st.markdown(f"✅ **{step_name.replace('_', ' ').title()}**")
                        elif status == 'running':
                            st.markdown(f"🔄 **{step_name.replace('_', ' ').title()}** (Processing...)")
                        else:
                            st.markdown(f"⏳ **{step_name.replace('_', ' ').title()}**")
                    
                    with col2:
                        if 'duration' in step:
                            st.markdown(f"*{step['duration']:.2f}s*")
                    
                    # Step output
                    if output and status == 'completed':
                        with st.expander(f"View {step_name} Output"):
                            try:
                                if isinstance(output, dict):
                                    st.json(output)
                                elif output:
                                    st.text(str(output))
                                else:
                                    st.info("No output available")
                            except Exception as e:
                                st.error(f"Error displaying output: {e}")
                                st.text(f"Raw output: {repr(output)}")
                    
                    # Search results
                    if 'search_results' in step:
                        results_count = len(step['search_results'])
                        st.markdown(f"🔍 Found {results_count} search results")
                    
                    # Judge evaluation
                    if 'judgment' in step and step['judgment'] is not None:
                        judgment = step['judgment']
                        if hasattr(judgment, 'is_sufficient'):
                            # Handle pydantic model objects
                            if judgment.is_sufficient:
                                st.success("⚖️ Judge: Information is sufficient")
                            else:
                                st.warning("⚖️ Judge: More information needed")
                                if hasattr(judgment, 'missing_information') and judgment.missing_information:
                                    st.markdown("Missing: " + ", ".join(judgment.missing_information))
                        elif isinstance(judgment, dict):
                            # Handle dictionary objects
                            if judgment.get('is_sufficient'):
                                st.success("⚖️ Judge: Information is sufficient")
                            else:
                                st.warning("⚖️ Judge: More information needed")
                                if judgment.get('missing_information'):
                                    st.markdown("Missing: " + ", ".join(judgment['missing_information']))
    
    def render_final_answer(self):
        """Render the final research answer."""
        if st.session_state.final_answer and st.session_state.workflow_state == "completed":
            st.header("📋 Legal Research Results")
            
            answer = st.session_state.final_answer
            
            # Main answer
            st.subheader("Answer")
            st.markdown(answer.get('answer', 'No answer provided'))
            
            # Key points
            if answer.get('key_points'):
                st.subheader("Key Points")
                for point in answer['key_points']:
                    st.markdown(f"• {point}")
            
            # Sources
            if answer.get('sources'):
                st.subheader("Sources")
                for source in answer['sources']:
                    st.markdown(f"• {source}")
            
            # Disclaimers
            if answer.get('disclaimers'):
                st.subheader("⚠️ Important Disclaimers")
                for disclaimer in answer['disclaimers']:
                    st.warning(disclaimer)
            
            # Confidence score
            if 'confidence' in answer:
                st.subheader("Confidence Score")
                confidence = answer['confidence']
                st.progress(confidence)
                st.markdown(f"**{confidence:.1%}** confidence in this answer")
    
    def render_trajectory_viewer(self):
        """Render trajectory analysis interface."""
        if getattr(st.session_state, 'show_trajectory_viewer', False):
            st.header("📊 Trajectory Analysis")
            
            try:
                viewer = ResultsViewer()
                runs = viewer.list_runs()
                
                if not runs:
                    st.info("No saved trajectory runs found")
                    return
                
                # Select run to view
                run_options = {
                    f"{run['run_id'][:8]}... - {run['start_time'][:19]} - {run['original_query'][:50]}...": run['run_id']
                    for run in runs[:10]  # Show latest 10
                }
                
                selected_run_key = st.selectbox(
                    "Select a run to analyze:",
                    list(run_options.keys())
                )
                
                if selected_run_key:
                    run_id = run_options[selected_run_key]
                    
                    # Tabs for different views
                    tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "🔄 Steps", "🤖 Model Calls", "👤 Human Inputs"])
                    
                    with tab1:
                        run_info = viewer.view_run(run_id)
                        if run_info:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Duration", run_info['duration'])
                                st.metric("Steps", run_info['steps_count'])
                            
                            with col2:
                                st.metric("Human Inputs", run_info['human_inputs_count'])
                                st.text(f"Query: {run_info['original_query']}")
                    
                    with tab2:
                        steps = viewer.view_run_steps(run_id)
                        if steps:
                            for step in steps:
                                st.markdown(f"**{step['node_name'].replace('_', ' ').title()}**")
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.text(f"Duration: {step['duration_seconds']:.2f}s")
                                with col2:
                                    st.text(f"Model calls: {step['model_calls_count']}")
                                st.markdown("---")
                    
                    with tab3:
                        interactions = viewer.view_model_interactions(run_id)
                        if interactions:
                            for interaction in interactions:
                                with st.expander(f"{interaction['step_name']} - {interaction['model_name']}"):
                                    st.text("Input:")
                                    st.code(interaction['input_prompt'])
                                    st.text("Output:")
                                    st.code(interaction['output_response'])
                    
                    with tab4:
                        human_inputs = viewer.view_human_inputs(run_id)
                        if human_inputs:
                            for key, value in human_inputs.items():
                                st.text(f"{key}: {value}")
                        else:
                            st.info("No human inputs recorded for this run")
                    
                    # Export button
                    if st.button("📥 Export Detailed JSON"):
                        try:
                            export_file = viewer.export_run(run_id)
                            st.success(f"Exported to: {export_file}")
                        except Exception as e:
                            st.error(f"Export failed: {e}")
            
            except Exception as e:
                st.error(f"Error loading trajectory data: {e}")
            
            # Close button
            if st.button("❌ Close Trajectory Viewer"):
                st.session_state.show_trajectory_viewer = False
                st.rerun()
    
    def start_research(self, query: str):
        """Start the legal research workflow."""
        if not st.session_state.lmars_system:
            st.error("System not initialized")
            return
        
        st.session_state.current_query = query
        st.session_state.workflow_state = "running"
        st.session_state.workflow_steps = []
        st.session_state.follow_up_questions = []
        st.session_state.final_answer = None
        
        # Start research in background
        self.execute_research_workflow(query)
    
    def execute_research_workflow(self, query: str):
        """Execute the research workflow with real-time updates."""
        try:
            config = {"configurable": {"thread_id": f"streamlit_session_{int(time.time())}"}}
            
            # Progress placeholder
            progress_placeholder = st.empty()
            step_placeholder = st.empty()
            
            step_count = 0
            for event in st.session_state.lmars_system.stream(query, config):
                step_count += 1
                current_step = event.get("current_step", "unknown")
                
                # Update progress
                with progress_placeholder:
                    st.info(f"🔄 Step {step_count}: {current_step.replace('_', ' ').title()}")
                
                # Handle follow-up questions
                if current_step == "follow_up_questions" and event.get("follow_up_questions"):
                    st.session_state.follow_up_questions = event["follow_up_questions"]
                    st.session_state.workflow_state = "paused"
                    with step_placeholder:
                        st.warning("⏸️ Workflow paused for user input")
                    st.rerun()
                    return
                
                # Store step information safely
                step_info = {
                    'step_name': current_step,
                    'status': 'completed',
                    'output': event.get("messages", [])[-1].content if event.get("messages") else "",
                    'search_results': event.get("search_results", []),
                    'judgment': event.get("judgment")
                }
                
                # Only append if this step isn't already recorded or allow multiple judge evaluations
                existing_step_names = [s.get('step_name') for s in st.session_state.workflow_steps]
                if current_step not in existing_step_names or current_step == "judge_evaluation":
                    st.session_state.workflow_steps.append(step_info)
                    
                st.session_state.current_run_id = event.get("run_id")
                
                # Check for final answer
                if event.get("final_answer"):
                    st.session_state.final_answer = {
                        'answer': event["final_answer"].answer,
                        'key_points': event["final_answer"].key_points,
                        'sources': event["final_answer"].sources,
                        'disclaimers': event["final_answer"].disclaimers,
                        'confidence': event["final_answer"].confidence
                    }
            
            # Complete workflow
            st.session_state.workflow_state = "completed"
            progress_placeholder.empty()
            with step_placeholder:
                st.success("✅ Research completed successfully!")
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Research failed: {e}")
            st.session_state.workflow_state = "idle"
            st.rerun()
    
    def continue_research(self, responses: Dict[str, str]):
        """Continue research after follow-up questions."""
        try:
            st.session_state.workflow_state = "running"
            
            config = {"configurable": {"thread_id": f"streamlit_session_{int(time.time())}"}}
            
            # Progress placeholder
            progress_placeholder = st.empty()
            
            step_count = len(st.session_state.workflow_steps)
            for event in st.session_state.lmars_system.continue_conversation(responses, config):
                step_count += 1
                current_step = event.get("current_step", "unknown")
                
                # Update progress
                with progress_placeholder:
                    st.info(f"🔄 Step {step_count}: {current_step.replace('_', ' ').title()}")
                
                # Store step information safely
                step_info = {
                    'step_name': current_step,
                    'status': 'completed',
                    'output': event.get("messages", [])[-1].content if event.get("messages") else "",
                    'search_results': event.get("search_results", []),
                    'judgment': event.get("judgment")
                }
                
                # Only append if this step isn't already recorded or allow multiple judge evaluations
                existing_step_names = [s.get('step_name') for s in st.session_state.workflow_steps]
                if current_step not in existing_step_names or current_step == "judge_evaluation":
                    st.session_state.workflow_steps.append(step_info)
                
                # Check for final answer
                if event.get("final_answer"):
                    st.session_state.final_answer = {
                        'answer': event["final_answer"].answer,
                        'key_points': event["final_answer"].key_points,
                        'sources': event["final_answer"].sources,
                        'disclaimers': event["final_answer"].disclaimers,
                        'confidence': event["final_answer"].confidence
                    }
            
            # Complete workflow
            st.session_state.workflow_state = "completed"
            progress_placeholder.empty()
            st.success("✅ Research completed successfully!")
            
            st.rerun()
            
        except Exception as e:
            st.error(f"Research continuation failed: {e}")
            st.session_state.workflow_state = "idle"
            st.rerun()
    
    def reset_session(self):
        """Reset the current session."""
        for key in ['current_query', 'workflow_steps', 'follow_up_questions', 
                   'final_answer', 'current_run_id']:
            if key in st.session_state:
                del st.session_state[key]
        
        st.session_state.workflow_state = "idle"
        if hasattr(st.session_state, 'show_trajectory_viewer'):
            del st.session_state.show_trajectory_viewer
    
    def run(self):
        """Main application runner."""
        try:
            # Render components
            self.render_header()
            self.render_sidebar()
            
            # Initialize system if needed
            if not st.session_state.lmars_system:
                self.initialize_system()
            
            # Main content area
            if getattr(st.session_state, 'show_trajectory_viewer', False):
                self.render_trajectory_viewer()
            else:
                self.render_query_input()
                self.render_follow_up_questions()
                self.render_workflow_progress()
                self.render_final_answer()
                
        except Exception as e:
            st.error(f"Application Error: {e}")
            st.info("Try refreshing the page or check the browser console for more details")
            
            # Show error details in an expander for debugging
            with st.expander("🐛 Error Details (for debugging)"):
                import traceback
                st.code(traceback.format_exc())


def main():
    """Main entry point for the Streamlit app."""
    app = LMarsStreamlitApp()
    app.run()


if __name__ == "__main__":
    main()