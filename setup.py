#!/usr/bin/env python3
"""
Setup script for nanoKimi

Installs dependencies and prepares the environment.
"""

import subprocess
import sys
import os


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"Processing: {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"SUCCESS: {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {description} failed:")
        print(f"Command: {command}")
        print(f"Error: {e.stderr}")
        return False


def main():
    print("=" * 60)
    print("nanoKimi Setup Script")
    print("Setting up the environment for Kimi-K2 training")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("ERROR: Python 3.8 or higher is required")
        print(f"Current version: {python_version.major}.{python_version.minor}")
        return False
    
    print(f"SUCCESS: Python version: {python_version.major}.{python_version.minor}")
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing Python dependencies"):
        return False
    
    # Create output directory
    os.makedirs("out", exist_ok=True)
    print("SUCCESS: Created output directory")
    
    # Prepare data
    print("\\nProcessing: Preparing Shakespeare dataset...")
    try:
        # Import and run data preparation
        sys.path.append('data')
        from prepare import prepare_shakespeare_data
        prepare_shakespeare_data()
        print("SUCCESS: Data preparation completed")
    except Exception as e:
        print(f"ERROR: Data preparation failed: {e}")
        return False
    
    print("\\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print("\\nNext steps:")
    print("1. Train a model: python train.py")
    print("2. Generate text: python generate.py --prompt 'To be or not to be'")
    print("3. Run benchmarks: python benchmark.py")
    print("\\nFor more options, use --help with any script.")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
