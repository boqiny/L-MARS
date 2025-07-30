"""
Utility functions for L-MARS Streamlit application.
"""
import streamlit as st
import time
import json
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
import hashlib


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable format."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.0f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        return f"{hours}h {remaining_minutes}m"


def format_timestamp(timestamp: str) -> str:
    """Format ISO timestamp to readable format."""
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp


def calculate_time_ago(timestamp: str) -> str:
    """Calculate time ago from timestamp."""
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        now = datetime.now(dt.tzinfo)
        diff = now - dt
        
        if diff.days > 0:
            return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
        elif diff.seconds > 3600:
            hours = diff.seconds // 3600
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif diff.seconds > 60:
            minutes = diff.seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        else:
            return "Just now"
    except:
        return "Unknown"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length with suffix."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def generate_session_id() -> str:
    """Generate a unique session ID."""
    timestamp = str(int(time.time() * 1000))
    hash_input = f"streamlit_session_{timestamp}_{st.session_state.get('user_id', 'anonymous')}"
    return hashlib.md5(hash_input.encode()).hexdigest()[:8]


def safe_json_serialize(obj: Any) -> str:
    """Safely serialize object to JSON string."""
    try:
        return json.dumps(obj, indent=2, default=str)
    except (TypeError, ValueError):
        return str(obj)


def parse_model_response(response: Any) -> Dict[str, Any]:
    """Parse and structure model response for display."""
    if hasattr(response, 'content'):
        content = response.content
    elif isinstance(response, dict):
        content = response.get('content', str(response))
    else:
        content = str(response)
    
    return {
        'content': content,
        'type': type(response).__name__,
        'length': len(str(content)),
        'timestamp': datetime.now().isoformat()
    }


def create_download_link(data: str, filename: str, mime_type: str = "text/plain") -> str:
    """Create a download link for data."""
    import base64
    
    b64_data = base64.b64encode(data.encode()).decode()
    href = f'<a href="data:{mime_type};base64,{b64_data}" download="{filename}">Download {filename}</a>'
    return href


class StreamlitLogger:
    """Simple logger for Streamlit apps."""
    
    def __init__(self):
        if 'app_logs' not in st.session_state:
            st.session_state.app_logs = []
    
    def log(self, level: str, message: str, details: Optional[Dict] = None):
        """Add a log entry."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level.upper(),
            'message': message,
            'details': details or {}
        }
        
        st.session_state.app_logs.append(entry)
        
        # Keep only last 100 logs
        if len(st.session_state.app_logs) > 100:
            st.session_state.app_logs = st.session_state.app_logs[-100:]
    
    def info(self, message: str, details: Optional[Dict] = None):
        """Log info message."""
        self.log('INFO', message, details)
    
    def warning(self, message: str, details: Optional[Dict] = None):
        """Log warning message."""
        self.log('WARNING', message, details)
    
    def error(self, message: str, details: Optional[Dict] = None):
        """Log error message."""
        self.log('ERROR', message, details)
    
    def get_logs(self, level: Optional[str] = None) -> List[Dict]:
        """Get logs, optionally filtered by level."""
        logs = st.session_state.app_logs
        
        if level:
            logs = [log for log in logs if log['level'] == level.upper()]
        
        return sorted(logs, key=lambda x: x['timestamp'], reverse=True)
    
    def render_logs(self, max_entries: int = 20):
        """Render logs in Streamlit."""
        logs = self.get_logs()[:max_entries]
        
        if not logs:
            st.info("No logs available")
            return
        
        for log in logs:
            level_colors = {
                'INFO': '🟢',
                'WARNING': '🟡', 
                'ERROR': '🔴'
            }
            
            color = level_colors.get(log['level'], '⚪')
            timestamp = format_timestamp(log['timestamp'])
            
            with st.expander(f"{color} {log['level']} - {log['message']} ({timestamp})"):
                st.text(f"Message: {log['message']}")
                if log['details']:
                    st.json(log['details'])


class ProgressTracker:
    """Track and display progress of multi-step operations."""
    
    def __init__(self, total_steps: int, operation_name: str = "Operation"):
        self.total_steps = total_steps
        self.current_step = 0
        self.operation_name = operation_name
        self.start_time = time.time()
        self.step_times = []
        
        # Create Streamlit containers
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        self.step_container = st.empty()
    
    def update(self, step_name: str, step_number: Optional[int] = None):
        """Update progress to next step."""
        if step_number is not None:
            self.current_step = step_number
        else:
            self.current_step += 1
        
        # Record timing
        current_time = time.time()
        self.step_times.append(current_time)
        
        # Calculate progress
        progress = min(self.current_step / self.total_steps, 1.0)
        self.progress_bar.progress(progress)
        
        # Update status
        elapsed = current_time - self.start_time
        estimated_total = elapsed * self.total_steps / max(self.current_step, 1)
        remaining = max(0, estimated_total - elapsed)
        
        status_msg = f"Step {self.current_step}/{self.total_steps}: {step_name}"
        if remaining > 0:
            status_msg += f" (Est. {format_duration(remaining)} remaining)"
        
        self.status_text.text(status_msg)
        
        # Update step details
        with self.step_container:
            if self.current_step > 1:
                avg_step_time = elapsed / (self.current_step - 1)
                st.text(f"Average step time: {format_duration(avg_step_time)}")
    
    def complete(self, final_message: str = "Completed successfully!"):
        """Mark operation as complete."""
        self.progress_bar.progress(1.0)
        total_time = time.time() - self.start_time
        self.status_text.success(f"{final_message} (Total time: {format_duration(total_time)})")
    
    def error(self, error_message: str):
        """Mark operation as failed."""
        total_time = time.time() - self.start_time
        self.status_text.error(f"Failed: {error_message} (After {format_duration(total_time)})")


def create_feedback_form():
    """Create a user feedback form."""
    with st.form("feedback_form"):
        st.subheader("📝 Feedback")
        
        feedback_type = st.selectbox(
            "Feedback Type",
            ["General", "Bug Report", "Feature Request", "Question"]
        )
        
        rating = st.select_slider(
            "How would you rate your experience?",
            options=[1, 2, 3, 4, 5],
            value=4,
            format_func=lambda x: "⭐" * x
        )
        
        feedback_text = st.text_area(
            "Your feedback:",
            placeholder="Please share your thoughts, suggestions, or report any issues..."
        )
        
        submitted = st.form_submit_button("Submit Feedback")
        
        if submitted and feedback_text:
            # In a real app, this would send feedback to a backend
            st.success("Thank you for your feedback! We appreciate your input.")
            
            # Log the feedback
            logger = StreamlitLogger()
            logger.info("Feedback submitted", {
                'type': feedback_type,
                'rating': rating,
                'feedback': feedback_text[:100] + "..." if len(feedback_text) > 100 else feedback_text
            })


def render_keyboard_shortcuts():
    """Display keyboard shortcuts help."""
    with st.expander("⌨️ Keyboard Shortcuts"):
        shortcuts = [
            ("Ctrl/Cmd + Enter", "Submit query or form"), 
            ("Ctrl/Cmd + R", "Reset session"),
            ("Ctrl/Cmd + S", "Save current state"),
            ("Ctrl/Cmd + /", "Toggle help"),
            ("Esc", "Close modals/popups")
        ]
        
        for shortcut, description in shortcuts:
            st.markdown(f"**{shortcut}:** {description}")


def check_system_health() -> Dict[str, Any]:
    """Check system health and return status."""
    import psutil
    import sys
    
    health = {
        'status': 'healthy',
        'checks': {},
        'warnings': [],
        'errors': []
    }
    
    try:
        # Memory usage
        memory = psutil.virtual_memory()
        health['checks']['memory_usage'] = f"{memory.percent}%"
        if memory.percent > 80:
            health['warnings'].append(f"High memory usage: {memory.percent}%")
        
        # Python version
        health['checks']['python_version'] = sys.version.split()[0]
        
        # Streamlit version
        health['checks']['streamlit_version'] = st.__version__
        
        # Session state size
        session_size = len(str(st.session_state))
        health['checks']['session_size'] = f"{session_size} chars"
        if session_size > 100000:
            health['warnings'].append("Large session state may affect performance")
        
    except Exception as e:
        health['errors'].append(f"Health check failed: {str(e)}")
        health['status'] = 'degraded'
    
    if health['warnings']:
        health['status'] = 'warning'
    if health['errors']:
        health['status'] = 'error'
    
    return health


def render_system_status():
    """Render system status information."""
    health = check_system_health()
    
    status_colors = {
        'healthy': '🟢',
        'warning': '🟡',
        'degraded': '🟠',
        'error': '🔴'
    }
    
    st.sidebar.markdown(f"**System Status:** {status_colors.get(health['status'], '⚪')} {health['status'].title()}")
    
    if health['warnings'] or health['errors']:
        with st.sidebar.expander("⚠️ System Issues"):
            for warning in health['warnings']:
                st.warning(warning)
            for error in health['errors']:
                st.error(error)
    
    # Detailed health info
    if st.sidebar.checkbox("Show System Details"):
        st.sidebar.json(health['checks'])