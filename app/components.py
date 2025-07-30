"""
Reusable Streamlit components for L-MARS application.
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any
import pandas as pd


def render_progress_chart(steps: List[Dict[str, Any]]) -> None:
    """Render a progress chart showing workflow steps and timing."""
    if not steps:
        return
    
    # Prepare data for timeline chart
    timeline_data = []
    start_time = 0
    
    for i, step in enumerate(steps):
        duration = step.get('duration', 1.0)
        timeline_data.append({
            'Step': step.get('step_name', f'Step {i+1}').replace('_', ' ').title(),
            'Start': start_time,
            'Duration': duration,
            'End': start_time + duration,
            'Status': step.get('status', 'completed')
        })
        start_time += duration
    
    df = pd.DataFrame(timeline_data)
    
    # Create Gantt-like chart
    fig = go.Figure()
    
    colors = {
        'completed': '#28a745',
        'running': '#ffc107',
        'pending': '#6c757d'
    }
    
    for i, row in df.iterrows():
        color = colors.get(row['Status'], '#17a2b8')
        
        fig.add_trace(go.Bar(
            name=row['Step'],
            y=[row['Step']],
            x=[row['Duration']],
            orientation='h',
            marker_color=color,
            text=f"{row['Duration']:.2f}s",
            textposition='middle right'
        ))
    
    fig.update_layout(
        title="Workflow Execution Timeline",
        xaxis_title="Time (seconds)",
        yaxis_title="Workflow Steps",
        height=max(300, len(steps) * 40),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_confidence_meter(confidence: float) -> None:
    """Render a confidence meter gauge."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Answer Confidence"},
        delta = {'reference': 80},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "yellow"},
                {'range': [80, 100], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


def render_search_results_summary(steps: List[Dict[str, Any]]) -> None:
    """Render a summary of search results across steps."""
    search_data = []
    
    for step in steps:
        if step.get('search_results'):
            search_data.append({
                'Step': step.get('step_name', 'Unknown').replace('_', ' ').title(),
                'Results': len(step['search_results']),
                'Sources': len(set(r.get('source', 'Unknown') for r in step['search_results']))
            })
    
    if not search_data:
        return
    
    df = pd.DataFrame(search_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(
            df, 
            x='Step', 
            y='Results',
            title="Search Results per Step"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(
            df, 
            x='Step', 
            y='Sources',
            title="Unique Sources per Step"
        )
        st.plotly_chart(fig, use_container_width=True)


def render_step_details(step: Dict[str, Any], step_index: int) -> None:
    """Render detailed information about a workflow step."""
    step_name = step.get('step_name', f'Step {step_index + 1}')
    
    with st.expander(f"📋 {step_name.replace('_', ' ').title()}", expanded=False):
        
        # Basic info
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Status", step.get('status', 'Unknown'))
        
        with col2:
            if 'duration' in step:
                st.metric("Duration", f"{step['duration']:.2f}s")
        
        with col3:
            if step.get('search_results'):
                st.metric("Results Found", len(step['search_results']))
        
        # Output
        if step.get('output'):
            st.subheader("Output")
            if isinstance(step['output'], str):
                st.text_area("Step Output", step['output'], height=150, disabled=True)
            else:
                st.json(step['output'])
        
        # Search results
        if step.get('search_results'):
            st.subheader("Search Results")
            for i, result in enumerate(step['search_results'][:3]):  # Show top 3
                with st.container():
                    st.markdown(f"**Result {i+1}:** {result.get('title', 'No title')}")
                    st.markdown(f"*Source: {result.get('source', 'Unknown')}*")
                    if result.get('content'):
                        st.text(result['content'][:200] + "..." if len(result['content']) > 200 else result['content'])
                    st.markdown("---")
        
        # Judge evaluation
        if step.get('judgment'):
            judgment = step['judgment']
            st.subheader("Judge Evaluation")
            
            if judgment.get('is_sufficient'):
                st.success("✅ Information deemed sufficient")
            else:
                st.warning("❌ Information deemed insufficient")
                
                if judgment.get('missing_information'):
                    st.markdown("**Missing Information:**")
                    for missing in judgment['missing_information']:
                        st.markdown(f"• {missing}")
                
                if judgment.get('suggested_refinements'):
                    st.markdown("**Suggested Refinements:**")
                    for refinement in judgment['suggested_refinements']:
                        st.markdown(f"• {refinement}")


def render_model_performance_stats(trajectory_data: Dict[str, Any]) -> None:
    """Render performance statistics for model calls."""
    if not trajectory_data.get('steps'):
        return
    
    model_stats = {}
    total_calls = 0
    total_duration = 0
    
    for step in trajectory_data['steps']:
        step_duration = step.get('duration_seconds', 0)
        total_duration += step_duration
        
        model_calls = step.get('model_calls', [])
        total_calls += len(model_calls)
        
        for call in model_calls:
            model_name = call.get('model_name', 'Unknown')
            if model_name not in model_stats:
                model_stats[model_name] = {
                    'calls': 0,
                    'tokens_used': 0,
                    'avg_duration': 0
                }
            
            model_stats[model_name]['calls'] += 1
            
            token_usage = call.get('token_usage', {})
            model_stats[model_name]['tokens_used'] += token_usage.get('total_tokens', 0)
    
    if not model_stats:
        return
    
    st.subheader("📊 Model Performance Statistics")
    
    # Overview metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Model Calls", total_calls)
    
    with col2:
        st.metric("Total Duration", f"{total_duration:.2f}s")
    
    with col3:
        avg_per_call = total_duration / total_calls if total_calls > 0 else 0
        st.metric("Avg per Call", f"{avg_per_call:.2f}s")
    
    # Model breakdown
    if len(model_stats) > 1:
        df = pd.DataFrame([
            {
                'Model': model,
                'Calls': stats['calls'],
                'Tokens': stats['tokens_used']
            }
            for model, stats in model_stats.items()
        ])
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(df, values='Calls', names='Model', title="Model Call Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(df, x='Model', y='Tokens', title="Token Usage by Model")
            st.plotly_chart(fig, use_container_width=True)


def render_status_badge(status: str) -> str:
    """Render a colored status badge."""
    badges = {
        "idle": "🟢 Ready",
        "running": "🟡 Processing",
        "paused": "🟠 Waiting",
        "completed": "✅ Complete",
        "error": "❌ Error"
    }
    
    return badges.get(status, "⚪ Unknown")


def render_copyable_code(code: str, language: str = "text") -> None:
    """Render code with a copy button."""
    st.code(code, language=language)
    
    if st.button("📋 Copy to Clipboard", key=f"copy_{hash(code)}"):
        # Note: Actual clipboard functionality would require additional setup
        st.success("Code copied! (Note: Clipboard functionality may require browser permissions)")


def render_legal_disclaimer() -> None:
    """Render standard legal disclaimer."""
    with st.expander("⚠️ Legal Disclaimer", expanded=False):
        st.warning("""
        **IMPORTANT LEGAL DISCLAIMER:**
        
        This system provides general legal information and research assistance only. 
        It does not constitute legal advice, and should not be relied upon as such.
        
        - This information may not be current or accurate
        - Laws vary by jurisdiction and change frequently  
        - Every legal situation is unique
        - You should consult with a qualified attorney for legal advice specific to your situation
        
        The creators of this system disclaim any liability for decisions made based on this information.
        """)


def render_export_options(run_id: str) -> None:
    """Render export options for trajectory data."""
    st.subheader("📥 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Export JSON", key=f"export_json_{run_id}"):
            st.info("JSON export functionality would be implemented here")
    
    with col2:
        if st.button("Export PDF", key=f"export_pdf_{run_id}"):
            st.info("PDF export functionality would be implemented here")
    
    with col3:
        if st.button("Export CSV", key=f"export_csv_{run_id}"):
            st.info("CSV export functionality would be implemented here")