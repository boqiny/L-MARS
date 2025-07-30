#!/usr/bin/env python3
"""
Launch script for L-MARS Streamlit application.
"""
import os
import sys
import subprocess
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("📁 Loaded environment variables from .env file")
except ImportError:
    print("⚠️  python-dotenv not installed. Install with: pip install python-dotenv")
except Exception as e:
    print(f"⚠️  Could not load .env file: {e}")


def main():
    """Launch the Streamlit application."""
    # Add project root to Python path
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # Set environment variables for the app
    os.environ['PYTHONPATH'] = str(project_root)
    
    # Path to the main app file
    app_file = project_root / "app" / "main.py"
    
    if not app_file.exists():
        print("❌ Error: Streamlit app file not found at", app_file)
        sys.exit(1)
    
    # Check for .env file
    env_file = project_root / ".env"
    if not env_file.exists():
        print("📝 No .env file found. Creating example .env file...")
        with open(env_file, 'w') as f:
            f.write("# L-MARS API Configuration\n")
            f.write("# Add your API keys below:\n\n")
            f.write("OPENAI_API_KEY=your_openai_api_key_here\n")
            f.write("# ANTHROPIC_API_KEY=your_anthropic_api_key_here\n")
        print(f"📝 Created example .env file at {env_file}")
        print("📝 Please edit the .env file with your actual API keys")
    
    # Check for required environment variables
    api_keys_available = (
        os.getenv("OPENAI_API_KEY") or 
        os.getenv("ANTHROPIC_API_KEY")
    )
    
    if not api_keys_available:
        print("⚠️  Warning: No API keys found in environment variables")
        print("   Set OPENAI_API_KEY or ANTHROPIC_API_KEY for full functionality")
        print("   The app will still launch but may have limited capabilities")
        print()
    
    # Launch Streamlit
    print("🚀 Starting L-MARS Streamlit Application...")
    print(f"📁 Project root: {project_root}")
    print(f"🎯 App file: {app_file}")
    print("🌐 Opening in browser...")
    print()
    
    try:
        # Run streamlit with the app file
        cmd = [
            sys.executable, "-m", "streamlit", "run", str(app_file),
            "--theme.base", "light",
            "--theme.primaryColor", "#1f77b4",
            "--theme.backgroundColor", "#ffffff",
            "--theme.secondaryBackgroundColor", "#f0f2f6",
            "--browser.gatherUsageStats", "false"
        ]
        
        subprocess.run(cmd, cwd=project_root)
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down L-MARS application...")
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()