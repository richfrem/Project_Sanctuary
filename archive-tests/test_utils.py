"""
Test utilities for Project Sanctuary.

Provides portable path computation functions that work across Windows, WSL, and Linux.
All paths are computed relative to file locations, never hardcoded.
"""

from pathlib import Path
from typing import Optional


def get_project_root() -> Path:
    """
    Get project root directory from any test file.
    
    This file is at: Project_Sanctuary/tests/test_utils.py
    So project root is one level up from this file's parent.
    
    Returns:
        Path to Project_Sanctuary root directory
    
    Example:
        >>> root = get_project_root()
        >>> assert (root / "README.md").exists()
    """
    # This file: Project_Sanctuary/tests/test_utils.py
    # Parent: Project_Sanctuary/tests/
    # Parent.parent: Project_Sanctuary/
    return Path(__file__).resolve().parent.parent


def get_test_data_dir() -> Path:
    """
    Get test data/fixtures directory.
    
    Returns:
        Path to tests/fixtures directory
    """
    return get_project_root() / "tests" / "fixtures"


def get_module_path(module_name: str) -> Path:
    """
    Get path to a specific module directory.
    
    Args:
        module_name: Name of module (e.g., "council_orchestrator", "mnemonic_cortex")
    
    Returns:
        Path to module directory
    
    Example:
        >>> orchestrator_path = get_module_path("council_orchestrator")
        >>> assert (orchestrator_path / "orchestrator").exists()
    """
    return get_project_root() / module_name


def get_file_relative_to_project(relative_path: str) -> Path:
    """
    Get absolute path to a file relative to project root.
    
    Args:
        relative_path: Path relative to project root (e.g., "01_PROTOCOLS/001_protocol.md")
    
    Returns:
        Absolute Path object
    
    Example:
        >>> config = get_file_relative_to_project("config/settings.json")
        >>> assert config.is_absolute()
    """
    return get_project_root() / relative_path


def ensure_test_dir_exists(dir_name: str) -> Path:
    """
    Ensure a test directory exists, create if needed.
    
    Args:
        dir_name: Directory name relative to tests/ (e.g., "fixtures", "temp")
    
    Returns:
        Path to directory
    """
    test_dir = get_project_root() / "tests" / dir_name
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir
