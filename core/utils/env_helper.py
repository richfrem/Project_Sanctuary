"""
Simple environment variable helper with proper fallback.

Provides consistent secret loading across Project Sanctuary with proper priority:
1. Environment variable (Windows → WSL via WSLENV)
2. .env file in project root
3. Error or None if not found

This ensures consistency with docs/WSL_SECRETS_CONFIGURATION.md
"""

import os
from typing import Optional
from pathlib import Path


def get_env_variable(key: str, required: bool = True) -> Optional[str]:
    """
    Get environment variable with proper fallback.
    
    Priority:
    1. Environment variable (Windows → WSL via WSLENV)
    2. .env file in project root
    3. Return None or raise error if not found
    
    Args:
        key: Environment variable name
        required: If True, raise error when not found
    
    Returns:
        Environment variable value or None
    
    Raises:
        ValueError: If required=True and variable not found
    
    Example:
        >>> from core.utils.env_helper import get_env_variable
        >>> token = get_env_variable("HUGGING_FACE_TOKEN", required=True)
    """
    # First, check environment (includes WSLENV passthrough from Windows)
    value = os.getenv(key)
    
    # Fallback to .env file if not in environment
    if not value:
        try:
            from dotenv import load_dotenv
            # Compute project root from this file's location
            # This file: Project_Sanctuary/core/utils/env_helper.py
            # Project root: ../../.. from this file
            project_root = Path(__file__).resolve().parent.parent.parent
            env_file = project_root / ".env"
            if env_file.exists():
                load_dotenv(env_file)
                value = os.getenv(key)
        except ImportError:
            # python-dotenv not installed, skip .env fallback
            pass
    
    # Handle missing required variables
    if required and not value:
        raise ValueError(
            f"Required environment variable not found: {key}\n"
            f"Please set this in Windows User Environment Variables.\n"
            f"See docs/WSL_SECRETS_CONFIGURATION.md for setup instructions."
        )
    
    return value
