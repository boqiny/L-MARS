#!/usr/bin/env python3
"""
LegalMind Streamlit Frontend

A comprehensive Streamlit frontend for the LegalMind legal research workflow,
featuring:
- Interactive workflow execution
- Real-time terminal output for all steps  
- Human-in-the-loop clarification UI
- Configuration management
- Session persistence
"""
import streamlit as st
import sys
import os
import json
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from io import StringIO
import contextlib

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Try importing LegalMind components
try:
    from lmars import (
        create_initial_state,
        optimized_legal_research_graph,
        ConfigManager,
        PRODUCTION_CONFIG,
        INTERACTIVE_CONFIG,
        DEVELOPMENT_CONFIG,
        __version__
    )
    
    # Try importing LangGraph components
    try:
        from langgraph.checkpoint.memory import MemorySaver
        from langgraph.types import Command
        LANGGRAPH_AVAILABLE = True
    except ImportError:
        LANGGRAPH_AVAILABLE = False
        st.warning("⚠️ LangGraph not installed - some features may be limited")
    
    LEGALMIND_AVAILABLE = True
except ImportError as e:
    LEGALMIND_AVAILABLE = False
    st.error(f"❌ LegalMind not available: {e}")


class TerminalCapture:
    """Capture and stream terminal output in real-time."""
    
    def __init__(self):
        self.output = []
        self.current_step = ""
    
    def write(self, text: str):
        """Capture output and add to terminal."""
        if text.strip():
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.output.append(f"[{timestamp}] {text.strip()}")
    
    def get_output(self) -> List[str]:
        """Get all captured output."""
        return self.output
    
    def clear(self):
        """Clear captured output."""
        self.output = []
    
    def add_step(self, step_name: str):
        """Add a step marker to the terminal."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.output.append(f"[{timestamp}] 🔄 Starting: {step_name}")
        self.current_step = step_name


class WorkflowManager:
    """Manages workflow execution with real-time monitoring."""
    
    def __init__(self):
        self.terminal = TerminalCapture()
        self.current_state = None
        self.workflow_status = "idle"
        self.checkpointer = None
        
        if LANGGRAPH_AVAILABLE:
            self.checkpointer = MemorySaver()
    
    def execute_workflow(self, query: str, config: Dict[str, Any], session_id: str) -> Dict[str, Any]:
        """Execute the workflow with real-time monitoring."""
        
        self.terminal.clear()
        self.terminal.add_step("Initializing workflow")
        
        try:
            # Create initial state
            initial_state = create_initial_state(
                query, 
                enable_human_clarification=config.get('enable_human_clarification', False)
            )
            
            self.terminal.add_step("Creating workflow graph")
            
            # Get the graph
            if self.checkpointer and config.get('enable_persistence', True):
                # Build a fresh graph with checkpointer
                from lmars.core.graph import build_legal_graph
                graph = build_legal_graph(checkpointer=self.checkpointer)
                graph_config = {"configurable": {"thread_id": session_id}}
                self.terminal.write(f"✅ Using persistent checkpointer (session: {session_id})")
            else:
                graph = optimized_legal_research_graph
                graph_config = {}
                self.terminal.write("ℹ️ Using in-memory execution (no persistence)")
            
            self.terminal.add_step("Executing workflow")
            self.workflow_status = "running"
            
            # Execute workflow with step monitoring
            result = self._execute_with_monitoring(graph, initial_state, graph_config)
            
            self.workflow_status = "completed"
            self.terminal.add_step("Workflow completed")
            
            return result
            
        except Exception as e:
            self.workflow_status = "error"
            self.terminal.write(f"❌ Error: {str(e)}")
            raise e
    
    def _execute_with_monitoring(self, graph, initial_state, config):
        """Execute workflow with step-by-step monitoring."""
        
        # Capture stdout to monitor agent outputs
        old_stdout = sys.stdout
        captured_output = StringIO()
        
        try:
            # Redirect stdout to capture agent prints
            sys.stdout = captured_output
            
            # Execute the workflow
            if LANGGRAPH_AVAILABLE and config:
                # Use streaming execution for real-time monitoring
                events = graph.stream(initial_state, config, stream_mode="values")
                
                result = None
                for event in events:
                    # Log each event
                    self.terminal.write(f"📊 Event keys: {list(event.keys())}")
                    
                    # Log specific state changes
                    if event.get('needs_clarification'):
                        self.terminal.write("🤔 Clarification needed")
                    if event.get('search_triggered'):
                        self.terminal.write(f"🔍 Search triggered: {event.get('search_reasoning', 'N/A')}")
                    if event.get('refined_sources'):
                        sources_preview = event['refined_sources'][:100] + "..." if len(event['refined_sources']) > 100 else event['refined_sources']
                        self.terminal.write(f"📚 Sources refined: {sources_preview}")
                    if event.get('final_answer'):
                        self.terminal.write("🎯 Final answer generated")
                    
                    result = event
                
                # Check for interrupts (human-in-the-loop)
                snapshot = graph.get_state(config)
                if snapshot.next:
                    self.terminal.write(f"⏸️ Workflow paused at: {snapshot.next}")
                    result['_workflow_interrupted'] = True
                    result['_next_steps'] = snapshot.next
                
                return result or initial_state
            
            else:
                # Fallback to simple execution
                result = graph.invoke(initial_state)
                self.terminal.write("✅ Workflow completed (simple execution)")
                return result
                
        finally:
            # Restore stdout and capture any output
            sys.stdout = old_stdout
            captured = captured_output.getvalue()
            
            # Add captured output to terminal
            if captured:
                for line in captured.split('\n'):
                    if line.strip():
                        self.terminal.write(line)
    
    def resume_workflow(self, session_id: str, human_input: str) -> Dict[str, Any]:
        """Resume interrupted workflow with human input."""
        
        if not self.checkpointer:
            raise Exception("Cannot resume - no checkpointer available")
        
        self.terminal.add_step("Resuming workflow with human input")
        
        try:
            # Create resume command
            human_command = Command(resume={"response": human_input})
            config = {"configurable": {"thread_id": session_id}}
            
            # Compile graph with checkpointer
            graph = optimized_legal_research_graph.compile(checkpointer=self.checkpointer)
            
            # Resume execution
            self.terminal.write(f"▶️ Resuming with input: {human_input[:50]}...")
            
            result = self._execute_with_monitoring(graph, human_command, config)
            
            self.terminal.add_step("Workflow resumed and completed")
            return result
            
        except Exception as e:
            self.terminal.write(f"❌ Resume error: {str(e)}")
            raise e


def init_session_state():
    """Initialize Streamlit session state."""
    
    if 'workflow_manager' not in st.session_state:
        st.session_state.workflow_manager = WorkflowManager()
    
    if 'session_id' not in st.session_state:
        st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"
    
    if 'workflow_result' not in st.session_state:
        st.session_state.workflow_result = None
    
    if 'workflow_config' not in st.session_state:
        st.session_state.workflow_config = {
            'enable_human_clarification': False,
            'enable_persistence': True,
            'mode': 'production'
        }
    
    if 'waiting_for_clarification' not in st.session_state:
        st.session_state.waiting_for_clarification = False


def render_header():
    """Render the app header."""
    
    st.set_page_config(
        page_title="LegalMind Research Assistant",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("⚖️ LegalMind Research Assistant")
    st.markdown("*AI-powered legal research with human-in-the-loop capabilities*")
    
    # Status indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_color = "🟢" if LEGALMIND_AVAILABLE else "🔴"
        st.metric("LegalMind", f"{status_color} {'Ready' if LEGALMIND_AVAILABLE else 'Error'}")
    
    with col2:
        lang_status = "🟢" if LANGGRAPH_AVAILABLE else "🟡"
        st.metric("LangGraph", f"{lang_status} {'Ready' if LANGGRAPH_AVAILABLE else 'Limited'}")
    
    with col3:
        workflow_status = st.session_state.workflow_manager.workflow_status
        status_colors = {"idle": "🔵", "running": "🟡", "completed": "🟢", "error": "🔴"}
        st.metric("Workflow", f"{status_colors.get(workflow_status, '⚪')} {workflow_status.title()}")
    
    with col4:
        st.metric("Session", f"🏷️ {st.session_state.session_id[:8]}")


def render_sidebar():
    """Render the configuration sidebar."""
    
    st.sidebar.header("⚙️ Configuration")
    
    # Workflow mode selection
    mode = st.sidebar.selectbox(
        "Workflow Mode",
        ["production", "interactive", "development"],
        index=0,
        help="Choose the workflow configuration mode"
    )
    
    # Update config based on mode
    if mode == "production":
        base_config = PRODUCTION_CONFIG
    elif mode == "interactive":
        base_config = INTERACTIVE_CONFIG
    else:
        base_config = DEVELOPMENT_CONFIG
    
    st.session_state.workflow_config['mode'] = mode
    
    st.sidebar.markdown("---")
    
    # Human-in-the-loop settings
    st.sidebar.subheader("🤔 Human-in-the-Loop")
    
    enable_human = st.sidebar.checkbox(
        "Enable Human Clarification",
        value=base_config.enable_human_clarification,
        help="Allow human input when clarification is needed"
    )
    st.session_state.workflow_config['enable_human_clarification'] = enable_human
    
    if enable_human and not LANGGRAPH_AVAILABLE:
        st.sidebar.warning("⚠️ LangGraph required for human-in-the-loop")
    
    # Persistence settings
    st.sidebar.subheader("💾 Persistence")
    
    enable_persistence = st.sidebar.checkbox(
        "Enable Session Persistence",
        value=base_config.enable_persistence and LANGGRAPH_AVAILABLE,
        disabled=not LANGGRAPH_AVAILABLE,
        help="Persist workflow state across interactions"
    )
    st.session_state.workflow_config['enable_persistence'] = enable_persistence
    
    # Advanced settings
    with st.sidebar.expander("🔧 Advanced Settings"):
        st.write(f"Search Confidence: {base_config.search_confidence_threshold}")
        st.write(f"Max Iterations: {base_config.max_workflow_iterations}")
        st.write(f"Compression Ratio: {base_config.target_compression_ratio}")
        st.write(f"Quality Gates: {base_config.enable_quality_gates}")
    
    # Session management
    st.sidebar.markdown("---")
    st.sidebar.subheader("🏷️ Session Management")
    
    if st.sidebar.button("🔄 New Session"):
        st.session_state.session_id = f"session_{uuid.uuid4().hex[:8]}"
        st.session_state.workflow_result = None
        st.session_state.waiting_for_clarification = False
        st.session_state.workflow_manager = WorkflowManager()
        st.rerun()
    
    st.sidebar.text(f"Current: {st.session_state.session_id}")


def render_terminal():
    """Render the terminal output component."""
    
    st.subheader("🖥️ Terminal Output")
    
    # Terminal container with fixed height and scrolling
    terminal_output = st.session_state.workflow_manager.terminal.get_output()
    
    if terminal_output:
        # Create a scrollable container
        terminal_text = "\n".join(terminal_output)
        st.code(terminal_text, language="bash")
        
        # Auto-scroll to bottom
        if st.session_state.workflow_manager.workflow_status == "running":
            st.empty()  # Force refresh for auto-scroll
    else:
        st.info("Terminal output will appear here during workflow execution...")
    
    # Clear terminal button
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("🗑️ Clear Terminal"):
            st.session_state.workflow_manager.terminal.clear()
            st.rerun()


def render_clarification_ui():
    """Render human-in-the-loop clarification UI."""
    
    if not st.session_state.waiting_for_clarification:
        return
    
    st.warning("🤔 **Human Clarification Required**")
    st.info("The workflow has paused and needs additional information to proceed.")
    
    # Show clarification request details
    result = st.session_state.workflow_result
    if result and result.get('clarification_request'):
        st.markdown("**What needs clarification:**")
        st.write(result['clarification_request'])
    
    # Clarification input
    clarification = st.text_area(
        "Please provide additional details:",
        height=100,
        placeholder="Provide specific details about your legal query..."
    )
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        if st.button("✅ Submit Clarification", type="primary"):
            if clarification.strip():
                try:
                    # Resume workflow with clarification
                    st.info("▶️ Resuming workflow with your clarification...")
                    result = st.session_state.workflow_manager.resume_workflow(
                        st.session_state.session_id,
                        clarification
                    )
                    
                    st.session_state.workflow_result = result
                    st.session_state.waiting_for_clarification = False
                    st.success("✅ Workflow resumed successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error resuming workflow: {str(e)}")
            else:
                st.error("Please provide clarification before submitting.")
    
    with col2:
        if st.button("❌ Skip Clarification"):
            # Use automatic clarification
            auto_clarification = "Please proceed with reasonable assumptions based on the original query."
            try:
                result = st.session_state.workflow_manager.resume_workflow(
                    st.session_state.session_id,
                    auto_clarification
                )
                st.session_state.workflow_result = result
                st.session_state.waiting_for_clarification = False
                st.success("✅ Workflow resumed with automatic clarification!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error resuming workflow: {str(e)}")


def render_main_interface():
    """Render the main research interface."""
    
    st.subheader("📝 Legal Research Query")
    
    # Query input
    query = st.text_area(
        "Enter your legal research question:",
        height=100,
        placeholder="Example: Can an LLC be sued in its own name under Delaware law?",
        disabled=st.session_state.workflow_manager.workflow_status == "running"
    )
    
    # Execute button
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if st.button(
            "🚀 Start Research",
            type="primary",
            disabled=not query.strip() or not LEGALMIND_AVAILABLE or st.session_state.workflow_manager.workflow_status == "running"
        ):
            try:
                with st.spinner("🔄 Executing legal research workflow..."):
                    # Execute workflow
                    result = st.session_state.workflow_manager.execute_workflow(
                        query,
                        st.session_state.workflow_config,
                        st.session_state.session_id
                    )
                    
                    st.session_state.workflow_result = result
                    
                    # Check if workflow was interrupted for clarification
                    if result.get('_workflow_interrupted'):
                        st.session_state.waiting_for_clarification = True
                        st.info("⏸️ Workflow paused for human clarification")
                    else:
                        st.success("✅ Research workflow completed!")
                    
                    st.rerun()
                    
            except Exception as e:
                st.error(f"❌ Workflow execution failed: {str(e)}")
    
    with col2:
        if st.button("⏹️ Stop", disabled=st.session_state.workflow_manager.workflow_status != "running"):
            st.session_state.workflow_manager.workflow_status = "idle"
            st.warning("⏹️ Workflow stopped")
            st.rerun()
    
    with col3:
        if st.button("🔄 Reset"):
            st.session_state.workflow_result = None
            st.session_state.waiting_for_clarification = False
            st.session_state.workflow_manager = WorkflowManager()
            st.success("🔄 Workflow reset")
            st.rerun()


def render_results():
    """Render workflow results."""
    
    if not st.session_state.workflow_result:
        return
    
    st.subheader("📊 Research Results")
    
    result = st.session_state.workflow_result
    
    # Result tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Final Answer", "🔍 Search Details", "📚 Sources", "🔧 Debug Info"])
    
    with tab1:
        if result.get('final_answer'):
            st.markdown("### Final Legal Analysis")
            st.write(result['final_answer'])
        else:
            st.info("Final answer not yet available")
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Search Triggered", "✅ Yes" if result.get('search_triggered') else "❌ No")
            if result.get('search_reasoning'):
                st.write("**Reasoning:**", result['search_reasoning'])
        
        with col2:
            quality_score = result.get('retrieval_quality_score', 0.0)
            st.metric("Quality Score", f"{quality_score:.2f}", delta=f"{quality_score - 0.5:.2f}")
        
        if result.get('clarified_query'):
            st.markdown("**Clarified Query:**")
            st.write(result['clarified_query'])
    
    with tab3:
        if result.get('refined_sources'):
            st.markdown("### Refined Legal Sources")
            st.write(result['refined_sources'])
        else:
            st.info("No sources retrieved")
    
    with tab4:
        st.markdown("### Debug Information")
        
        debug_info = {
            "Session ID": st.session_state.session_id,
            "Workflow Status": st.session_state.workflow_manager.workflow_status,
            "Iteration Count": result.get('iteration_count', 0),
            "Judge Satisfied": result.get('judge_satisfied', False),
            "Configuration": st.session_state.workflow_config
        }
        
        st.json(debug_info)


def main():
    """Main Streamlit app."""
    
    # Initialize session state
    init_session_state()
    
    # Render UI components
    render_header()
    render_sidebar()
    
    # Main layout - two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        render_main_interface()
        render_clarification_ui()
    
    with col2:
        render_terminal()
    
    # Results section (full width)
    render_results()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using [LegalMind](https://github.com/your-repo/legalmind) • "
        "[LangGraph](https://langchain-ai.github.io/langgraph/) • "
        "[Streamlit](https://streamlit.io/)"
    )


if __name__ == "__main__":
    if not LEGALMIND_AVAILABLE:
        st.error("❌ LegalMind not available. Please install the package and configure API keys.")
        st.stop()
    
    main()