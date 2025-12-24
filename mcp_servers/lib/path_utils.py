"""
Path utilities for Project Sanctuary.
"""

import os
from pathlib import Path

def find_project_root() -> str:
    """Find the project root by ascending from the current script's directory."""
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
