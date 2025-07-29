#!/usr/bin/env python3
"""
LegalMind Streamlit App Launcher

Provides an easy way to run the Streamlit frontend with proper configuration.
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def check_dependencies():
    """Check if required dependencies are installed."""
    dependencies_ok = True
    
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__} found")
    except ImportError:
        print("❌ Streamlit not found. Install with:")
        print("   pip install -e .[streamlit]")
        dependencies_ok = False
    
    try:
        import lmars
        version = getattr(lmars, '__version__', 'unknown')
        print(f"✅ LegalMind {version} found")
    except ImportError as e:
        print(f"❌ LegalMind not found. Import error: {e}")
        print("   Install with: pip install -e .")
        dependencies_ok = False
    except Exception as e:
        print(f"⚠️  LegalMind found but version check failed: {e}")
        print("✅ LegalMind available (version check failed)")
    
    return dependencies_ok


def setup_environment():
    """Setup environment variables for the app."""
    # Set default Streamlit configuration
    os.environ.setdefault("STREAMLIT_THEME_BASE", "light")
    os.environ.setdefault("STREAMLIT_THEME_PRIMARY_COLOR", "#FF6B35")
    os.environ.setdefault("STREAMLIT_THEME_BACKGROUND_COLOR", "#FFFFFF")
    os.environ.setdefault("STREAMLIT_THEME_SECONDARY_BACKGROUND_COLOR", "#F0F2F6")
    os.environ.setdefault("STREAMLIT_THEME_TEXT_COLOR", "#262730")
    
    # LegalMind configuration
    os.environ.setdefault("LEGALMIND_ENABLE_HUMAN_CLARIFICATION", "false")
    os.environ.setdefault("LEGALMIND_ENABLE_PERSISTENCE", "true")
    os.environ.setdefault("LEGALMIND_MAX_ITERATIONS", "3")
    
    print("🔧 Environment configured")


def run_streamlit(port=8501, host="localhost", dev_mode=False):
    """Run the Streamlit app."""
    
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    app_file = script_dir / "streamlit_app.py"
    
    if not app_file.exists():
        print(f"❌ Streamlit app not found at {app_file}")
        return False
    
    # Build streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_file),
        "--server.port", str(port),
        "--server.address", host,
        "--browser.gatherUsageStats", "false",
    ]
    
    if dev_mode:
        cmd.extend([
            "--server.runOnSave", "true",
            "--server.fileWatcherType", "poll"
        ])
    
    print(f"🚀 Starting Streamlit app at http://{host}:{port}")
    print(f"   Command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit: {e}")
        return False
    
    return True


def main():
    """Main launcher function."""
    parser = argparse.ArgumentParser(
        description="Launch LegalMind Streamlit Frontend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_streamlit.py                    # Run on default port 8501
  python run_streamlit.py --port 8080       # Run on custom port
  python run_streamlit.py --dev             # Run in development mode
  python run_streamlit.py --host 0.0.0.0    # Allow external connections
        """
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8501,
        help="Port to run the app on (default: 8501)"
    )
    
    parser.add_argument(
        "--host", "-H",
        default="localhost",
        help="Host to bind to (default: localhost)"
    )
    
    parser.add_argument(
        "--dev", "-d",
        action="store_true",
        help="Run in development mode with auto-reload"
    )
    
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check dependencies without running the app"
    )
    
    args = parser.parse_args()
    
    print("⚖️ LegalMind Streamlit Frontend Launcher")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("\n💡 Installation instructions:")
        print("1. Install LegalMind: pip install -e .")
        print("2. Install Streamlit extras: pip install -e .[streamlit]")
        print("3. Set up API keys in .env file")
        return 1
    
    if args.check_deps:
        print("✅ All dependencies are available")
        return 0
    
    # Setup environment
    setup_environment()
    
    # Run the app
    success = run_streamlit(
        port=args.port,
        host=args.host,
        dev_mode=args.dev
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())