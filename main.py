#!/usr/bin/env python3
"""
L-MARS Main Entry Point
Default: Simple Mode for quick legal research
"""
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lmars.cli import main

if __name__ == "__main__":
    sys.exit(main())