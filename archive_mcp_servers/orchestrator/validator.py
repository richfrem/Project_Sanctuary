#!/usr/bin/env python3
"""
Orchestrator Validator
=====================================

Purpose:
    Validation logic for Orchestrator Operations.
    Enforces safety rules (e.g. protected paths) and git protocols/convention compliance.

Layer: Validation (Logic)

Key Classes:
    - OrchestratorValidator: Main safety logic
        - __init__(project_root, config_path)
        - validate_path(path)
        - validate_git_operation(files, message, push)
        - validate_cognitive_task(output_path)
        - _load_config(path)
"""

import os
import re
import json
from pathlib import Path
from typing import List, Dict, Any
from .models import ValidationResult

class OrchestratorValidator:
    """
    Class: OrchestratorValidator
    Purpose: Validates commands against safety rules and git safety protocols.
    """
    
    def __init__(self, project_root: str, config_path: str = None):
        self.project_root = Path(project_root).resolve()
        
        # Load config
        if config_path:
            self.config = self._load_config(config_path)
        else:
            # Default config path relative to project root .agent/config/mcp_config.json if possible, 
            # OR local config folder. Original code looked in parent/config.
            # We will try to look in standard locations.
            # Local config fallback:
            local_config = Path(__file__).parent / "config" / "mcp_config.json"
            if local_config.exists():
                self.config = self._load_config(str(local_config))
            else:
                self.config = {}
            
        self.safety_config = self.config.get("safety", {})
        self.protected_paths = self.safety_config.get("protected_paths", [])
        self.allowed_extensions = set(self.safety_config.get("allowed_extensions", []))
        
        # Prohibited patterns for git commands
        self.prohibited_patterns = [
            r"git\s+reset\s+--hard",
            r"git\s+push\s+(-f|--force)",
            r"git\s+rebase",
            r"rm\s+-rf",
        ]

    def _load_config(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    #============================================
    # Method: validate_path
    # Purpose: Validate that a file path is safe to write to.
    # Args:
    #   path: Relative path string
    # Returns: ValidationResult
    #============================================
    def validate_path(self, path: str) -> ValidationResult:
        """Validate that a file path is safe to write to."""
        try:
            # Resolve absolute path
            if os.path.isabs(path):
                abs_path = Path(path).resolve()
            else:
                abs_path = (self.project_root / path).resolve()
            
            # Check if path is within project root
            if not str(abs_path).startswith(str(self.project_root)):
                return ValidationResult(False, f"Path traversal detected: {path}", "DANGEROUS")
            
            # Check protected paths
            try:
                rel_path = abs_path.relative_to(self.project_root)
            except ValueError:
                # Should be covered by startswith check, but safe guard
                return ValidationResult(False, f"Path outside project root: {path}", "DANGEROUS")

            for protected in self.protected_paths:
                if str(rel_path).startswith(protected):
                    return ValidationResult(False, f"Cannot modify protected path: {rel_path}", "DANGEROUS")
            
            # Check extension (if restricted list exists)
            if self.allowed_extensions and abs_path.suffix not in self.allowed_extensions:
                 return ValidationResult(False, f"File extension not allowed: {abs_path.suffix}", "MODERATE")

            return ValidationResult(True, risk_level="SAFE")
            
        except Exception as e:
            return ValidationResult(False, f"Path validation error: {str(e)}", "DANGEROUS")

    #============================================
    # Method: validate_git_operation
    # Purpose: Validate git commit operation.
    # Args:
    #   files: List of files
    #   message: Commit message
    #   push: Push flag
    # Returns: ValidationResult
    #============================================
    def validate_git_operation(self, files: List[str], message: str, push: bool) -> ValidationResult:
        """Validate git commit operation against safety rules."""
        
        # Validate all files
        for file_path in files:
            res = self.validate_path(file_path)
            if not res.valid:
                return res
        
        # Validate commit message format (conventional commits)
        conventional_commit_pattern = r"^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\(.+\))?: .+"
        if not re.match(conventional_commit_pattern, message):
            return ValidationResult(
                valid=False,
                reason="Commit message must follow conventional commit format (e.g., 'feat(scope): description')",
                risk_level="MODERATE"
            )
        
        # Check for prohibited patterns in message (injection check)
        for pattern in self.prohibited_patterns:
            if re.search(pattern, message):
                 return ValidationResult(False, "Commit message contains prohibited patterns", "DANGEROUS")

        # Check push risk
        if push:
             return ValidationResult(True, risk_level="MODERATE")
        
        return ValidationResult(valid=True, risk_level="SAFE")

    #============================================
    # Method: validate_cognitive_task
    # Purpose: Validate cognitive task parameters.
    # Args:
    #   output_path: Path for output artifact
    # Returns: ValidationResult
    #============================================
    def validate_cognitive_task(self, output_path: str) -> ValidationResult:
        """Validate cognitive task parameters."""
        return self.validate_path(output_path)
