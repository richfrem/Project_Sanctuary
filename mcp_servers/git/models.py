#!/usr/bin/env python3
"""
Git Models
=====================================

Purpose:
    Data definitions for Git operations.
    Defines Pydantic models for status, branch info, and tool requests.

Layer: Data (DTOs)

Key Models:
    # Internal / Enums
    - BranchInfo: Name and current status
    - RemoteInfo: Upstream, ahead/behind counts
    - GitStatus: Full repository state (staged, modified, etc.)

    # MCP Requests
    - GitAddRequest
    - GitDiffRequest
    - GitFinishFeatureRequest
    - GitLogRequest
    - GitPushFeatureRequest
    - GitSmartCommitRequest
    - GitStartFeatureRequest
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

#============================================
# Class: BranchInfo
# Purpose: Pydantic model for git branch information.
#============================================
class BranchInfo(BaseModel):
    name: str
    current: bool

#============================================
# Class: RemoteInfo
# Purpose: Pydantic model for git remote status (ahead/behind).
#============================================
class RemoteInfo(BaseModel):
    upstream: Optional[str] = None
    ahead: int = 0
    behind: int = 0

#============================================
# Class: GitStatus
# Purpose: Comprehensive git status model.
# Role: Data Transfer Object
#============================================
class GitStatus(BaseModel):
    branch: str
    staged: List[str] = Field(default_factory=list)
    modified: List[str] = Field(default_factory=list)
    untracked: List[str] = Field(default_factory=list)
    local_branches: List[BranchInfo] = Field(default_factory=list)
    feature_branches: List[str] = Field(default_factory=list)
    remote: RemoteInfo = Field(default_factory=RemoteInfo)
    is_clean: bool = True

#============================================
# FastMCP Request Models
#============================================

class GitAddRequest(BaseModel):
    files: Optional[List[str]] = Field(None, description="List of files to stage. If None, stages all changes.")

class GitDiffRequest(BaseModel):
    cached: bool = Field(False, description="Show staged changes only")
    file_path: Optional[str] = Field(None, description="Specific file to diff")

class GitFinishFeatureRequest(BaseModel):
    branch_name: str = Field(..., description="The name of the feature branch to finish")
    force: bool = Field(False, description="Force finish even if not merged")

class GitLogRequest(BaseModel):
    max_count: int = Field(10, description="Number of commits to show")
    oneline: bool = Field(False, description="Use one-line format")

class GitPushFeatureRequest(BaseModel):
    force: bool = Field(False, description="Force push to origin")
    no_verify: bool = Field(False, description="Bypass pre-push hooks")

class GitSmartCommitRequest(BaseModel):
    message: str = Field(..., description="Commit message following project standards")

class GitStartFeatureRequest(BaseModel):
    task_id: str = Field(..., description="Task identifier (e.g., '144')")
    description: str = Field(..., description="Brief feature description")
