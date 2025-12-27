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
from mcp_servers.lib.path_utils import find_project_root


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
        >>> from mcp_servers.lib.env_helper import get_env_variable
        >>> token = get_env_variable("HUGGING_FACE_TOKEN", required=True)
    """
    # First, check environment (includes WSLENV passthrough from Windows)
    value = os.getenv(key)
    
    # Fallback to .env file if not in environment
    if not value:
        try:
            from dotenv import load_dotenv
            # Use centralized path utility
            project_root = Path(find_project_root())
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

def load_env() -> bool:
    """
    Explicitly load .env file from project root.
    
    Returns:
        True if loaded, False otherwise
    """
    try:
        from dotenv import load_dotenv
        project_root = Path(find_project_root())
        env_file = project_root / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            return True
    except ImportError:
        pass
    return False
