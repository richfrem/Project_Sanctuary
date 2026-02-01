#!/usr/bin/env python3
"""
Config Validator
=====================================

Purpose:
    Validation logic for Config Operations.
    Enforces path security (jail) and file validity.

Layer: Validation (Logic)

Key Classes:
    - ConfigValidator: Main safety logic
        - __init__(config_dir)
        - validate_path(filename) -> Path
"""

from pathlib import Path

class ConfigValidator:
    """
    Class: ConfigValidator
    Purpose: Enforces security and safety constraints on config operations.
    """
    
    def __init__(self, config_dir: Path):
        self.config_dir = config_dir

    #============================================
    # Method: validate_path
    # Purpose: Ensure filename resolves to a path within config directory.
    # Args:
    #   filename: Name of the config file
    # Returns: Resolved Path object
    # Throws: ValueError if path violation
    #============================================
    def validate_path(self, filename: str) -> Path:
        """Validate that the file path is within the config directory."""
        # Resolve the full path
        file_path = (self.config_dir / filename).resolve()
        
        # Check if the resolved path starts with the config directory path
        if not str(file_path).startswith(str(self.config_dir)):
            raise ValueError(f"Security Error: Path '{filename}' is outside config directory")
            
        return file_path
