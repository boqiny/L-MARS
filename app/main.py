"""
L-MARS Legal Research System - Streamlit Frontend (Simplified)
Interactive web interface using the new two-mode workflow system.
"""
import streamlit as st
import sys
import os
from typing import Dict, Any
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from lmars import create_workflow, WorkflowConfig


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
        if 'workflow' not in st.session_state:
            st.session_state.workflow = None
        
        if 'current_mode' not in st.session_state:
            st.session_state.current_mode = "simple"
        
        if 'processing' not in st.session_state:
            st.session_state.processing = False
        
        if 'last_result' not in st.session_state:
            st.session_state.last_result = None
        
        if 'config' not in st.session_state:
            st.session_state.config = WorkflowConfig()
    
    def render_header(self):
        """Render the application header."""
        st.title("🏛️ L-MARS Legal Research System")
        st.markdown("*Simplified Two-Mode Legal Research: Simple (fast) or Multi-Turn (thorough)*")
        
        # Mode selector
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            mode = st.radio(
                "Select Mode:",
                ["simple", "multi_turn"],
                format_func=lambda x: "⚡ Simple Mode (Fast)" if x == "simple" else "🔄 Multi-Turn Mode (Thorough)",
                horizontal=True,
                key="mode_selector"
            )
            
            if mode != st.session_state.current_mode:
                st.session_state.current_mode = mode
                st.session_state.workflow = None
        
        with col3:
            if st.button("🔄 Reset"):
                self.reset_session()
                st.rerun()
    
    def render_sidebar(self):
        """Render the sidebar with configuration."""
        st.sidebar.header("⚙️ Configuration")
        
        # Model Selection
        st.sidebar.subheader("Model Settings")
        
        llm_model = st.sidebar.selectbox(
            "LLM Model",
            ["openai:gpt-4o", "openai:gpt-4", "openai:gpt-3.5-turbo"],
            index=0
        )
        
        if st.session_state.current_mode == "multi_turn":
            judge_model = st.sidebar.selectbox(
                "Judge Model (Multi-Turn)",
                ["openai:gpt-4o", "openai:gpt-4", "Same as LLM"],
                index=2
            )
            
            max_iterations = st.sidebar.slider(
                "Max Iterations",
                min_value=1,
                max_value=5,
                value=3
            )
        else:
            judge_model = None
            max_iterations = 3
        
        # Update config
        st.session_state.config = WorkflowConfig(
            mode=st.session_state.current_mode,
            llm_model=llm_model,
            judge_model=None if judge_model == "Same as LLM" else judge_model,
            max_iterations=max_iterations,
            enable_tracking=False
        )
        
        # API Key Status
        st.sidebar.subheader("🔑 API Keys")
        openai_key = bool(os.getenv("OPENAI_API_KEY"))
        serper_key = bool(os.getenv("SERPER_API_KEY"))
        
        if openai_key:
            st.sidebar.success("✅ OpenAI API Key")
        else:
            st.sidebar.error("❌ OpenAI API Key Missing")
        
        if serper_key:
            st.sidebar.success("✅ Serper API Key")
        else:
            st.sidebar.warning("⚠️ Serper API Key (Optional)")
        
        if not openai_key:
            st.sidebar.info("Add OPENAI_API_KEY to your .env file")
        
        # Mode Info
        st.sidebar.subheader("ℹ️ Current Mode")
        if st.session_state.current_mode == "simple":
            st.sidebar.info(
                "**Simple Mode**\n\n"
                "• Single-turn search\n"
                "• Fast responses\n"
                "• Best for straightforward questions"
            )
        else:
            st.sidebar.info(
                "**Multi-Turn Mode**\n\n"
                "• Iterative refinement\n"
                "• Follow-up questions\n"
                "• Judge evaluation\n"
                "• Best for complex queries"
            )
    
    def render_query_interface(self):
        """Render the main query interface."""
        st.header("💬 Ask Your Legal Question")
        
        # Query input
        query = st.text_area(
            "Enter your legal question:",
            height=100,
            placeholder="e.g., Can an F1 student work remotely for a US company while studying?",
            disabled=st.session_state.processing
        )
        
        col1, col2 = st.columns([1, 4])
        
        with col1:
            if st.button(
                "🚀 Research", 
                disabled=(not query.strip() or st.session_state.processing),
                type="primary"
            ):
                self.execute_research(query)
        
        with col2:
            if st.session_state.processing:
                st.info("🔄 Processing your query...")
    
    def execute_research(self, query: str):
        """Execute the research workflow."""
        st.session_state.processing = True
        
        try:
            # Initialize workflow if needed
            if not st.session_state.workflow:
                st.session_state.workflow = create_workflow(
                    mode=st.session_state.config.mode,
                    llm_model=st.session_state.config.llm_model,
                    judge_model=st.session_state.config.judge_model,
                    max_iterations=st.session_state.config.max_iterations,
                    enable_tracking=False
                )
            
            # Show progress
            with st.spinner(f"Researching using {st.session_state.current_mode.replace('_', ' ').title()} Mode..."):
                result = st.session_state.workflow.run(query)
            
            # Handle multi-turn follow-up questions
            if result.get("needs_user_input") and result.get("follow_up_questions"):
                self.handle_followup_questions(query, result)
            else:
                st.session_state.last_result = result
                self.display_results(result)
        
        except Exception as e:
            st.error(f"❌ Error: {e}")
        
        finally:
            st.session_state.processing = False
    
    def handle_followup_questions(self, original_query: str, result: Dict[str, Any]):
        """Handle follow-up questions in multi-turn mode."""
        st.subheader("❓ Clarification Needed")
        st.info("Please answer these questions to help me provide better assistance:")
        
        with st.form("followup_form"):
            responses = {}
            
            for i, question in enumerate(result.get("follow_up_questions", []), 1):
                st.markdown(f"**{i}. {question.question}**")
                st.caption(f"_{question.reason}_")
                response = st.text_input(f"Answer {i}:", key=f"q_{i}")
                if response:
                    responses[f"question_{i}"] = response
                st.markdown("---")
            
            if st.form_submit_button("Submit Answers", type="primary"):
                if responses:
                    with st.spinner("Processing with your answers..."):
                        final_result = st.session_state.workflow.run(original_query, responses)
                        st.session_state.last_result = final_result
                        self.display_results(final_result)
                else:
                    st.warning("Please provide at least one answer")
    
    def display_results(self, result: Dict[str, Any]):
        """Display the research results."""
        st.header("📋 Research Results")
        
        if not result.get("final_answer"):
            st.warning("No results found")
            return
        
        answer = result["final_answer"]
        
        # Main answer
        st.subheader("📌 Answer")
        st.markdown(answer.answer)
        
        # Key points
        if answer.key_points:
            st.subheader("🔑 Key Points")
            for point in answer.key_points:
                st.markdown(f"• {point}")
        
        # Sources
        if answer.sources:
            st.subheader("📚 Sources")
            for source in answer.sources:
                st.markdown(f"• {source}")
        
        # Metadata
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Confidence", f"{answer.confidence:.1%}")
        
        with col2:
            st.metric("Mode", result.get("mode", "unknown").replace("_", " ").title())
        
        with col3:
            if result.get("iterations"):
                st.metric("Iterations", result["iterations"])
        
        # Disclaimers
        if answer.disclaimers:
            with st.expander("⚠️ Legal Disclaimers"):
                for disclaimer in answer.disclaimers:
                    st.warning(disclaimer)
    
    def reset_session(self):
        """Reset the current session."""
        st.session_state.workflow = None
        st.session_state.last_result = None
        st.session_state.processing = False
    
    def run(self):
        """Main application runner."""
        # Render components
        self.render_header()
        self.render_sidebar()
        
        # Check API keys
        if not os.getenv("OPENAI_API_KEY"):
            st.error("🔑 Please set OPENAI_API_KEY in your .env file to continue")
            st.stop()
        
        # Main interface
        self.render_query_interface()
        
        # Display last results if available
        if st.session_state.last_result and not st.session_state.processing:
            self.display_results(st.session_state.last_result)


def main():
    """Main entry point for the Streamlit app."""
    app = LMarsStreamlitApp()
    app.run()


if __name__ == "__main__":
    main()