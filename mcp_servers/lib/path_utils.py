"""
Path utilities for Project Sanctuary.
"""

import os
from pathlib import Path

def find_project_root() -> str:
    """Find the project root by ascending from the current script's directory.
    
    Checks in order:
    1. PROJECT_ROOT environment variable (for container environments)
    2. .git folder ascending from current file
    3. .git folder in current working directory
    """
    # Check for explicit PROJECT_ROOT first (containers, etc.)
    project_root = os.getenv("PROJECT_ROOT")
    if project_root and os.path.isdir(project_root):
        return project_root
    
    current_path = os.path.abspath(os.path.dirname(__file__))
    while True:
        if '.git' in os.listdir(current_path):
            return current_path
        parent_path = os.path.dirname(current_path)
        if parent_path == current_path:
            # Fallback to current working directory if .git not found (e.g. in some container envs)
            cwd = os.getcwd()
            if '.git' in os.listdir(cwd):
                return cwd
            raise FileNotFoundError("Could not find the project root (.git folder).")
        current_path = parent_path
