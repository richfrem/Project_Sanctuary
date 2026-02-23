#!/usr/bin/env python3
"""
Git Validator
=====================================

Purpose:
    Validation logic for Git operations.
    Enforces Protocol 101 (Clean State) and Protocol 122 (Poka-Yoke).

Layer: Validation (Logic)

Key Classes:
    - GitValidator: Main safety logic
        - validate_clean_state(status)
        - validate_feature_branch_context(current_branch, operation)
        - validate_one_feature_rule(branch_name, existing_features)
        - validate_poka_yoke(staged_files, diff_getter)
"""

from typing import List, Optional
from mcp_servers.git.models import GitStatus

class ValidationError(Exception):
    """Custom exception schema for validation failures."""
    def __init__(self, message: str, remediation: Optional[str] = None):
        super().__init__(message)
        self.remediation = remediation

class GitValidator:
    """
    Enforces safety rules on Git operations.
    - Protocol 101: Functional Coherence (Clean state before critical ops)
    - Protocol 122: Poka-Yoke (High-risk file audits)
    """
    
    HIGH_RISK_FILES = [".gitignore", ".env", ".env.local", "Dockerfile", "package.json"]

    #============================================
    # Method: validate_clean_state
    # Purpose: Enforce Pillar 4: Pre-Execution Verification.
    # Args:
    #   status: GitStatus object
    # Raises: ValidationError if working directory is not clean
    #============================================
    #============================================
    # Method: validate_clean_state
    # Purpose: Enforce Pillar 4: Pre-Execution Verification.
    # Args:
    #   status: GitStatus object
    # Raises: ValidationError if working directory is not clean
    #============================================
    def validate_clean_state(self, status: GitStatus) -> None:
        if not status.is_clean:
            raise ValidationError(
                message=f"Working directory is not clean ("
                        f"Modified: {len(status.modified)}, "
                        f"Staged: {len(status.staged)}, "
                        f"Untracked: {len(status.untracked)}).",
                remediation="Commit or stash changes before proceeding."
            )

    #============================================
    # Method: validate_feature_branch_context
    # Purpose: Enforce that operations happen on feature branches.
    # Args:
    #   current_branch: Name of current branch
    #   operation: Name of operation being performed
    #   allow_main: Check bypass flag
    # Raises: ValidationError if on main branch without override
    #============================================
    #============================================
    # Method: validate_feature_branch_context
    # Purpose: Enforce that operations happen on feature branches.
    # Args:
    #   current_branch: Name of current branch
    #   operation: Name of operation being performed
    #   allow_main: Check bypass flag
    # Raises: ValidationError if on main branch without override
    #============================================
    def validate_feature_branch_context(self, current_branch: str, operation: str, allow_main: bool = False) -> None:
        if allow_main:
            return
            
        if current_branch == "main":
            raise ValidationError(
                message=f"SAFETY ERROR: Cannot perform '{operation}' on 'main' branch.",
                remediation="Switch to a feature branch using `git_start_feature`."
            )

    #============================================
    # Method: validate_one_feature_rule
    # Purpose: Enforce One Feature Rule.
    # Args:
    #   branch_name: Proposed new branch name
    #   existing_features: List of existing feature branches
    # Raises: ValidationError if feature branch limit exceeded
    #============================================
    #============================================
    # Method: validate_one_feature_rule
    # Purpose: Enforce One Feature Rule.
    # Args:
    #   branch_name: Proposed new branch name
    #   existing_features: List of existing feature branches
    # Raises: ValidationError if feature branch limit exceeded
    #============================================
    def validate_one_feature_rule(self, branch_name: str, existing_features: List[str]) -> None:
        if len(existing_features) > 0 and branch_name not in existing_features:
             raise ValidationError(
                message=f"One Feature Rule: Cannot create '{branch_name}'. Existing feature branch(es) detected: {', '.join(existing_features)}.",
                remediation="Finish the current feature branch first."
            )

    #============================================
    # Method: validate_poka_yoke
    # Purpose: Enforce Protocol 122: Audit high-risk files.
    # Args:
    #   staged_files: List of files being committed
    #   diff_getter: Function to retrieve file diffs
    # Raises: ValidationError if audit fails
    #============================================
    #============================================
    # Method: validate_poka_yoke
    # Purpose: Enforce Protocol 122: Audit high-risk files.
    # Args:
    #   staged_files: List of files being committed
    #   diff_getter: Function to retrieve file diffs
    # Raises: ValidationError if audit fails
    #============================================
    def validate_poka_yoke(self, staged_files: List[str], diff_getter) -> None:
        high_risk_staged = [f for f in staged_files if any(f.endswith(hr) for hr in self.HIGH_RISK_FILES)]
        
        if not high_risk_staged:
            return

        import sys
        print(f"[POKA-YOKE] High-risk files detected in staging: {high_risk_staged}", file=sys.stderr)
        
        for file_path in high_risk_staged:
            diff_output = diff_getter(file_path)
            
            # Check 1: Content Deletion
            lines_added = diff_output.count('\n+') - 1
            lines_removed = diff_output.count('\n-') - 1
            
            if lines_removed > 0 and lines_removed > lines_added * 2:
                raise ValidationError(
                    message=f"POKA-YOKE BLOCKED: High-risk file '{file_path}' has significant content removal.",
                    remediation="Review the diff carefully. This may indicate accidental clearing."
                )
            
            # Check 2: Secrets
            secret_patterns = ["API_KEY=", "SECRET=", "PASSWORD=", "TOKEN=", "aws_secret"]
            for pattern in secret_patterns:
                if pattern.lower() in diff_output.lower():
                     raise ValidationError(
                        message=f"POKA-YOKE BLOCKED: Potential secret detected in '{file_path}' (Pattern: {pattern}).",
                        remediation="Remove secrets before committing."
                    )
