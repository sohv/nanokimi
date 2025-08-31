#!/usr/bin/env python3
"""
Main benchmark script for nanoKimi - can be run from project root
"""

import os
import sys

# Add src and other directories to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.join(project_root, 'data'))
sys.path.insert(0, os.path.join(project_root, 'config'))

# Import and run the benchmark script
if __name__ == "__main__":
    from scripts.benchmark import main
    main()
