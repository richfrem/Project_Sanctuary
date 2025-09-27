"""
Core Utilities (core/utils.py)

This module provides essential utility functions used across the Mnemonic Cortex application.
These functions handle environment setup and path resolution to ensure reliable operation.

Role in RAG Pipeline:
- find_project_root(): Dynamically locates the project root by searching for the .git directory.
  This allows scripts to be run from any location within the project structure.
- setup_environment(): Loads environment variables from the .env file in the mnemonic_cortex directory.
  Ensures configuration (like DB_PATH and SOURCE_DOCUMENT_PATH) is available to all components.

Dependencies:
- Standard library: os for path operations.
- python-dotenv: For loading environment variables from .env files.
- Project structure: Relies on the presence of a .git directory at the project root.

These utilities are foundational and used by both ingestion and query pipelines.
"""

import os
from dotenv import load_dotenv

def find_project_root():
    """Find the project root by ascending from the current script's directory."""
    current_path = os.path.abspath(os.path.dirname(__file__))
    while True:
        if '.git' in os.listdir(current_path):
            return current_path
        parent_path = os.path.dirname(current_path)
        if parent_path == current_path:
            raise FileNotFoundError("Could not find the project root (.git folder).")
        current_path = parent_path

def setup_environment(project_root):
    """Load environment variables from the .env file in the mnemonic_cortex directory."""
    dotenv_path = os.path.join(project_root, 'mnemonic_cortex', '.env')
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path=dotenv_path)
        return True
    print(f"Warning: .env file not found at {dotenv_path}")
    return False