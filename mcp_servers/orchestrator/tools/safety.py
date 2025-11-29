import os
import re
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

class ValidationResult:
    def __init__(self, valid: bool, reason: str = "", risk_level: str = "SAFE"):
        self.valid = valid
        self.reason = reason
        self.risk_level = risk_level

class SafetyValidator:
    """Validates commands against safety rules and git safety protocols."""
    
    def __init__(self, project_root: str, config_path: str = None):
        self.project_root = Path(project_root).resolve()
        
        # Load config
        if config_path:
            self.config = self._load_config(config_path)
        else:
            # Default config path relative to this file
            config_path = Path(__file__).parent.parent / "config" / "mcp_config.json"
            self.config = self._load_config(str(config_path))
            
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
        except Exception as e:
            print(f"Warning: Could not load config from {path}: {e}")
            return {}

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
            rel_path = abs_path.relative_to(self.project_root)
            for protected in self.protected_paths:
                if str(rel_path).startswith(protected):
                    return ValidationResult(False, f"Cannot modify protected path: {rel_path}", "DANGEROUS")
            
            # Check extension
            if self.allowed_extensions and abs_path.suffix not in self.allowed_extensions:
                 return ValidationResult(False, f"File extension not allowed: {abs_path.suffix}", "MODERATE")

            return ValidationResult(True, risk_level="SAFE")
            
        except Exception as e:
            return ValidationResult(False, f"Path validation error: {str(e)}", "DANGEROUS")

    def validate_git_operation(self, files: List[str], message: str, push: bool) -> ValidationResult:
        """Validate git commit operation against safety rules."""
        
        # Validate all files
        for file_path in files:
            res = self.validate_path(file_path)
            if not res.valid:
                return res
        
        # Validate commit message format (conventional commits)
        # Regex for: type(scope): description or type: description
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

        # Check if we're on main branch (requires extra caution)
        # Note: In a real implementation, we'd check the current branch via git
        # For now, we assume push=True is risky if not verified
        if push:
             # We allow push if it's explicitly requested, but mark it as MODERATE risk
             # The tool implementation should decide whether to block it based on user approval settings
             return ValidationResult(True, risk_level="MODERATE")
        
        return ValidationResult(valid=True, risk_level="SAFE")

    def validate_cognitive_task(self, output_path: str) -> ValidationResult:
        """Validate cognitive task parameters."""
        return self.validate_path(output_path)
