#============================================
# mcp_servers/git/server.py
# Purpose: MCP Server for Git Operations.
#          Provides tools for branching, committing, pushing, and finishing features.
# Role: Protocol 101/128 Enforcement
# Used as: Main service entry point for the sanctuary_git cluster.
#============================================

import os
import sys
import json
from typing import List, Optional
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

# Local/Library Imports
from mcp_servers.lib.env_helper import get_env_variable
from mcp_servers.lib.path_utils import find_project_root
from mcp_servers.lib.logging_utils import setup_mcp_logging
from mcp_servers.git.operations import GitOperations
from .models import (
    GitAddRequest,
    GitDiffRequest,
    GitFinishFeatureRequest,
    GitLogRequest,
    GitPushFeatureRequest,
    GitSmartCommitRequest,
    GitStartFeatureRequest
)

# 1. Initialize Logging
logger = setup_mcp_logging("sanctuary_git")

# 2. Initialize FastMCP with Sanctuary Metadata
mcp = FastMCP(
    "project_sanctuary.version_control.git",
    instructions="""
    Use this server to manage git version control operations.
    Follow Protocol 101: Always pull before starting, and never commit to main.
    Use `git_get_status` frequently to verify the repository state.
    """
)

# 3. Initialize Operations
PROJECT_ROOT = find_project_root()
REPO_PATH = get_env_variable("REPO_PATH", required=False) or str(PROJECT_ROOT)
BASE_DIR = get_env_variable("GIT_BASE_DIR", required=False)
git_ops = GitOperations(REPO_PATH, base_dir=BASE_DIR)

#============================================
# Standardized Tool Implementations
#============================================

@mcp.tool()
async def git_get_safety_rules() -> str:
    """
    Get the unbreakable Git safety rules (Protocol 101).
    """
    return """
    🛡️ GIT SAFETY PRIMER: UNBREAKABLE RULES FOR AGENTS 🛡️
    1. SYNCHRONIZATION FIRST: Pull main before starting.
    2. MAIN IS PROTECTED: Never commit to main.
    3. SERIAL PROCESSING: One feature branch at a time.
    4. STATE VERIFICATION: Check status before acting.
    5. DESTRUCTIVE ACTION GATE: No force pushes without approval.
    6. NO GHOST EDITS: Verify branch before editing.
    """

@mcp.tool()
async def git_get_status() -> str:
    """
    Get the current status of the repository, including staged and modified files.
    """
    try:
        status = git_ops.status()
        return (
            f"Branch: {status.branch}\n"
            f"Staged Files: {', '.join(status.staged) if status.staged else 'None'}\n"
            f"Modified Files: {', '.join(status.modified) if status.modified else 'None'}\n"
            f"Untracked Files: {', '.join(status.untracked) if status.untracked else 'None'}"
        )
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        raise ToolError(f"Status check failed: {str(e)}")

@mcp.tool()
async def git_add(request: GitAddRequest) -> str:
    """
    Stage files for commit.
    """
    try:
        status = git_ops.status()
        if status.branch == "main":
            raise ToolError("ERROR: Cannot stage on main branch.")
            
        git_ops.add(request.files)
        return f"Successfully staged changes."
    except Exception as e:
        logger.error(f"Failed to stage files: {e}")
        raise ToolError(f"Add failed: {str(e)}")

@mcp.tool()
async def git_diff(request: GitDiffRequest) -> str:
    """
    Show changes in the working directory or staged area.
    """
    try:
        diff = git_ops.diff(cached=request.cached, file_path=request.file_path)
        return diff if diff else "No changes detected."
    except Exception as e:
        logger.error(f"Failed to get diff: {e}")
        raise ToolError(f"Diff failed: {str(e)}")

@mcp.tool()
async def git_log(request: GitLogRequest) -> str:
    """
    Show commit history for the current branch.
    """
    try:
        return git_ops.log(max_count=request.max_count, oneline=request.oneline)
    except Exception as e:
        logger.error(f"Failed to get log: {e}")
        raise ToolError(f"Log failed: {str(e)}")

@mcp.tool()
async def git_smart_commit(request: GitSmartCommitRequest) -> str:
    """
    Commit staged files with automatic Protocol 101 v3.0 enforcement.
    """
    try:
        status = git_ops.status()
        if status.branch == "main":
            raise ToolError("ERROR: Cannot commit directly to main. Use git_start_feature.")
        if not status.branch.startswith("feature/"):
            raise ToolError(f"ERROR: Invalid branch '{status.branch}'. Use feature/ format.")
            
        staged_files = git_ops.get_staged_files()
        if not staged_files:
            raise ToolError("ERROR: No files staged for commit.")
            
        commit_hash = git_ops.commit(request.message)
        return f"Successfully committed changes. Hash: {commit_hash}"
    except Exception as e:
        logger.error(f"Smart commit failed: {e}")
        raise ToolError(f"Commit failed: {str(e)}")

@mcp.tool()
async def git_start_feature(request: GitStartFeatureRequest) -> str:
    """
    Start a new feature branch following project standards.
    """
    try:
        return git_ops.start_feature(request.task_id, request.description)
    except Exception as e:
        logger.error(f"Failed to start feature: {e}")
        raise ToolError(f"Branch creation failed: {str(e)}")

@mcp.tool()
async def git_push_feature(request: GitPushFeatureRequest) -> str:
    """
    Push the current feature branch to origin.
    """
    try:
        current_branch = git_ops.get_current_branch()
        if current_branch == "main":
            raise ToolError("ERROR: Cannot push main directly.")
            
        git_ops.push("origin", current_branch, force=request.force, no_verify=request.no_verify)
        commit_hash = git_ops.get_commit_hash("HEAD")
        return f"Successfully pushed {current_branch} (Hash: {commit_hash[:8]})"
    except Exception as e:
        logger.error(f"Push failed: {e}")
        raise ToolError(f"Push failed: {str(e)}")

@mcp.tool()
async def git_finish_feature(request: GitFinishFeatureRequest) -> str:
    """
    Clean up a feature branch after it has been merged.
    """
    try:
        return git_ops.finish_feature(request.branch_name, force=request.force)
    except Exception as e:
        logger.error(f"Failed to finish feature: {e}")
        raise ToolError(f"Branch deletion failed: {str(e)}")

#============================================
# Main Execution Entry Point
#============================================

if __name__ == "__main__":
    # Dual-mode support:
    # 1. If PORT is set -> Run as SSE (Gateway Mode)
    # 2. If PORT is NOT set -> Run as Stdio (Local/CLI Mode)
    port_env = get_env_variable("PORT", required=False)
    transport = "sse" if port_env else "stdio"
    
    if transport == "sse":
        port = int(port_env) if port_env else 8003
        mcp.run(port=port, transport=transport)
    else:
        mcp.run(transport=transport)
