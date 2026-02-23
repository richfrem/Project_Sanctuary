#!/usr/bin/env python3
"""
Code Validator
=====================================

Purpose:
    Validation logic for Code Operations.
    Enforces path security (Jail) and Protocol 122 (Poka-Yoke) for safe writes.

Layer: Validation (Logic)

Key Classes:
    - CodeValidator: Main safety logic
        - __init__(project_root)
        - validate_path(path)
        - validate_safe_write(file_path, new_content)
"""

from pathlib import Path
from typing import List
from .models import HIGH_RISK_FILES

#============================================
# Class: CodeValidator
# Purpose: Enforces security and safety constraints on file operations.
#============================================
class CodeValidator:
    
    def __init__(self, project_root: Path):
        self.project_root = project_root

    #============================================
    # Method: validate_path
    # Purpose: Ensure path is within the project root (Jail).
    # Args:
    #   path: Relative path string
    # Returns: Resolved Path object
    # Throws: ValueError if path violation
    #============================================
    def validate_path(self, path: str) -> Path:
        """Validate that the path is within the project directory."""
        # Clean path and resolve relative to project root
        file_path = (self.project_root / path).resolve()
        
        if not str(file_path).startswith(str(self.project_root)):
            raise ValueError(f"Security Error: Path '{path}' is outside project directory")
            
        return file_path

    #============================================
    # Method: validate_safe_write
    # Purpose: Protocol 122 Enforcer - Prevent accidental data loss in high-risk files.
    # Args:
    #   file_path: Target Path object
    #   new_content: Content to write
    # Throws: ValueError on safety violation
    #============================================
    def validate_safe_write(self, file_path: Path, new_content: str) -> None:
        """
        Poka-Yoke: Enforces Protocol 122 (Read-Modify-Merge) for high-risk files.
        Blocks writes where new content is <50% of original content size.
        """
        # Only check if it's a high risk file
        if not any(file_path.name == f for f in HIGH_RISK_FILES):
            return

        if not file_path.exists():
            return
            
        try:
            original_content = file_path.read_text(encoding='utf-8')
        except Exception:
            # If we can't read original, be cautious but allow if it might be binary (though these are usually text)
            # For this specific list (env, gitignore, json, dockerfile), they are text.
            return

        # Content Loss Prevention
        if original_content and len(new_content) < (len(original_content) * 0.5):
            raise ValueError(
                f"POKA-YOKE BLOCKED: Write to high-risk file '{file_path.name}' rejected. "
                f"New content ({len(new_content)} chars) is <50% of original ({len(original_content)} chars). "
                f"This indicates a probable accidental overwrite. Use explicit merge logic."
            )
