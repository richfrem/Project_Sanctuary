#============================================
# mcp_servers/gateway/clusters/sanctuary_git/server.py
# Purpose: Sanctuary Git Cluster - Dual-Transport Entry Point
# Role: Interface Layer (Cluster Node)
# Status: ADR-066 v1.3 Compliant (SSEServer for Gateway, FastMCP for STDIO)
# Used by: Gateway Fleet (SSE) and Claude Desktop (STDIO)
#============================================

import os
import sys
import logging
from typing import List, Optional

# Local/Library Imports
from mcp_servers.lib.env_helper import get_env_variable
from mcp_servers.lib.path_utils import find_project_root
from mcp_servers.lib.logging_utils import setup_mcp_logging

# Setup Logging
logger = setup_mcp_logging("project_sanctuary.sanctuary_git")

# Configuration
PROJECT_ROOT = get_env_variable("PROJECT_ROOT", required=False) or find_project_root()
REPO_PATH = get_env_variable("REPO_PATH", required=False) or str(PROJECT_ROOT)
BASE_DIR = get_env_variable("GIT_BASE_DIR", required=False)

_git_ops = None

def get_ops():
    global _git_ops
    if _git_ops is None:
        from mcp_servers.git.operations import GitOperations
        _git_ops = GitOperations(REPO_PATH, base_dir=BASE_DIR)
    return _git_ops

def check_requirements() -> Optional[str]:
    req_file = os.path.join(REPO_PATH, "REQUIREMENTS.env")
    if not os.path.exists(req_file):
        return None
    try:
        with open(req_file, 'r') as f:
            pass
    except Exception as e:
        return f"Failed to verify requirements: {str(e)}"
    return None


#============================================
# Tool Schema Definitions (for SSEServer)
#============================================
SMART_COMMIT_SCHEMA = {
    "type": "object",
    "properties": {
        "message": {"type": "string", "description": "Commit message"}
    },
    "required": ["message"]
}

ADD_SCHEMA = {
    "type": "object",
    "properties": {
        "files": {"type": "array", "items": {"type": "string"}, "description": "Files to stage"}
    }
}

PUSH_FEATURE_SCHEMA = {
    "type": "object",
    "properties": {
        "force": {"type": "boolean", "description": "Force push"},
        "no_verify": {"type": "boolean", "description": "Skip pre-push hooks"}
    }
}

START_FEATURE_SCHEMA = {
    "type": "object",
    "properties": {
        "task_id": {"type": "integer", "description": "Task ID number"},
        "description": {"type": "string", "description": "Brief description"}
    },
    "required": ["task_id", "description"]
}

FINISH_FEATURE_SCHEMA = {
    "type": "object",
    "properties": {
        "branch_name": {"type": "string", "description": "Branch to finish"},
        "force": {"type": "boolean", "description": "Force delete"}
    },
    "required": ["branch_name"]
}

DIFF_SCHEMA = {
    "type": "object",
    "properties": {
        "cached": {"type": "boolean", "description": "Show staged changes"},
        "file_path": {"type": "string", "description": "Specific file"}
    }
}

LOG_SCHEMA = {
    "type": "object",
    "properties": {
        "max_count": {"type": "integer", "description": "Max commits"},
        "oneline": {"type": "boolean", "description": "One line per commit"}
    }
}

EMPTY_SCHEMA = {"type": "object", "properties": {}}


#============================================
# SSE Transport Implementation (Gateway Mode)
# Migrated to @sse_tool decorator pattern per ADR-076
#============================================
def run_sse_server(port: int):
    """Run using SSEServer for Gateway compatibility (ADR-066 v1.3)."""
    from mcp_servers.lib.sse_adaptor import SSEServer, sse_tool
    
    server = SSEServer("sanctuary_git", version="1.0.0")
    ops = get_ops()
    
    @sse_tool(
        name="git_smart_commit",
        description="Commit with automated Protocol 101 checks.",
        schema=SMART_COMMIT_SCHEMA
    )
    def git_smart_commit(message: str):
        status = ops.status()
        if status.branch == "main":
            return "ERROR: Cannot commit directly to main. Use git_start_feature."
        if not status.branch.startswith("feature/"):
            return f"ERROR: Invalid branch '{status.branch}'. Use feature/ format."
        staged = ops.get_staged_files()
        if not staged:
            return "ERROR: No files staged."
        commit_hash = ops.commit(message)
        return f"Commit successful. Hash: {commit_hash}"
    
    @sse_tool(
        name="git_get_safety_rules",
        description="Return Protocol 101 safety rules.",
        schema=EMPTY_SCHEMA
    )
    def git_get_safety_rules():
        return """🛡️ GIT SAFETY PRIMER: UNBREAKABLE RULES FOR AGENTS 🛡️
1. SYNCHRONIZATION FIRST: Pull main before starting.
2. MAIN IS PROTECTED: Never commit to main.
3. SERIAL PROCESSING: One feature branch at a time.
4. STATE VERIFICATION: Check status before acting.
5. DESTRUCTIVE ACTION GATE: No force pushes without approval.
6. NO GHOST EDITS: Verify branch before editing."""
    
    @sse_tool(
        name="git_get_status",
        description="Get standard git status.",
        schema=EMPTY_SCHEMA
    )
    def git_get_status():
        status = ops.status()
        return (
            f"Branch: {status.branch}\n"
            f"Staged Files: {', '.join(status.staged)}\n"
            f"Modified Files: {', '.join(status.modified)}\n"
            f"Untracked Files: {', '.join(status.untracked)}"
        )
    
    @sse_tool(
        name="git_add",
        description="Stage files for commit.",
        schema=ADD_SCHEMA
    )
    def git_add(files: List[str] = None):
        status = ops.status()
        if status.branch == "main":
            return "ERROR: Cannot stage on main branch."
        if not status.branch.startswith("feature/"):
            return f"ERROR: Invalid branch '{status.branch}'."
        ops.add(files)
        if files:
            return f"Staged {len(files)} file(s) on {status.branch}"
        return f"Staged all changes on {status.branch}"
    
    @sse_tool(
        name="git_push_feature",
        description="Push feature branch to origin.",
        schema=PUSH_FEATURE_SCHEMA
    )
    def git_push_feature(force: bool = False, no_verify: bool = False):
        current = ops.get_current_branch()
        if current == "main":
            return "ERROR: Cannot push main directly."
        if not current.startswith("feature/"):
            return f"ERROR: Invalid branch '{current}'."
        try:
            ops.push("origin", current, force=force, no_verify=no_verify)
        except RuntimeError as e:
            if "git-lfs" in str(e):
                ops.push("origin", current, force=force, no_verify=True)
            else:
                raise
        local_hash = ops.get_commit_hash("HEAD")
        pr_url = f"https://github.com/richfrem/Project_Sanctuary/pull/new/{current}"
        return f"Verified push to {current} (Hash: {local_hash[:8]}).\nLink: {pr_url}"
    
    @sse_tool(
        name="git_start_feature",
        description="Start a new feature branch.",
        schema=START_FEATURE_SCHEMA
    )
    def git_start_feature(task_id: int, description: str):
        req_error = check_requirements()
        if req_error:
            return req_error
        return ops.start_feature(task_id, description)
    
    @sse_tool(
        name="git_finish_feature",
        description="Finish feature (cleanup/delete).",
        schema=FINISH_FEATURE_SCHEMA
    )
    def git_finish_feature(branch_name: str, force: bool = False):
        return ops.finish_feature(branch_name, force=force)
    
    @sse_tool(
        name="git_diff",
        description="Show changes (diff).",
        schema=DIFF_SCHEMA
    )
    def git_diff(cached: bool = False, file_path: str = None):
        diff = ops.diff(cached=cached, file_path=file_path)
        return diff if diff else "No changes."
    
    @sse_tool(
        name="git_log",
        description="Show commit history.",
        schema=LOG_SCHEMA
    )
    def git_log(max_count: int = 10, oneline: bool = True):
        return ops.log(max_count=max_count, oneline=oneline)
    
    # Auto-register all decorated tools (ADR-076)
    server.register_decorated_tools(locals())
    
    logger.info(f"Starting SSEServer on port {port} (Gateway Mode)")
    server.run(port=port, transport="sse")


#============================================
# STDIO Transport Implementation (Local Mode)
#============================================
def run_stdio_server():
    """Run using FastMCP for local development (Claude Desktop)."""
    from fastmcp import FastMCP
    from fastmcp.exceptions import ToolError
    from mcp_servers.git.models import (
        GitSmartCommitRequest, GitAddRequest, GitPushFeatureRequest,
        GitStartFeatureRequest, GitFinishFeatureRequest, GitDiffRequest,
        GitLogRequest
    )
    
    mcp = FastMCP(
        "sanctuary_git",
        instructions="""
        Sanctuary Git Cluster.
        - specialized in automated feature branch workflows.
        - enforces Protocol 101 safety rules for agentic git operations.
        """
    )
    
    @mcp.tool()
    async def git_smart_commit(request: GitSmartCommitRequest) -> str:
        """Commit with automated Protocol 101 checks."""
        try:
            ops = get_ops()
            status = ops.status()
            if status.branch == "main":
                return "ERROR: Cannot commit directly to main. Use git_start_feature."
            if not status.branch.startswith("feature/"):
                return f"ERROR: Invalid branch '{status.branch}'. Use feature/ format."
            staged = ops.get_staged_files()
            if not staged:
                return "ERROR: No files staged."
            commit_hash = ops.commit(request.message)
            return f"Commit successful. Hash: {commit_hash}"
        except Exception as e:
            raise ToolError(f"Commit failed: {str(e)}")
    
    @mcp.tool()
    async def git_get_safety_rules() -> str:
        """Return Protocol 101 safety rules."""
        return """🛡️ GIT SAFETY PRIMER: UNBREAKABLE RULES FOR AGENTS 🛡️
1. SYNCHRONIZATION FIRST: Pull main before starting.
2. MAIN IS PROTECTED: Never commit to main.
3. SERIAL PROCESSING: One feature branch at a time.
4. STATE VERIFICATION: Check status before acting.
5. DESTRUCTIVE ACTION GATE: No force pushes without approval.
6. NO GHOST EDITS: Verify branch before editing."""
    
    @mcp.tool()
    async def git_get_status() -> str:
        """Get standard git status."""
        try:
            status = get_ops().status()
            return (
                f"Branch: {status.branch}\n"
                f"Staged Files: {', '.join(status.staged)}\n"
                f"Modified Files: {', '.join(status.modified)}\n"
                f"Untracked Files: {', '.join(status.untracked)}"
            )
        except Exception as e:
            raise ToolError(f"Status check failed: {str(e)}")
    
    @mcp.tool()
    async def git_add(request: GitAddRequest) -> str:
        """Stage files for commit."""
        try:
            ops = get_ops()
            status = ops.status()
            if status.branch == "main":
                return "ERROR: Cannot stage on main branch."
            if not status.branch.startswith("feature/"):
                return f"ERROR: Invalid branch '{status.branch}'."
            ops.add(request.files)
            if request.files:
                return f"Staged {len(request.files)} file(s) on {status.branch}"
            return f"Staged all changes on {status.branch}"
        except Exception as e:
            raise ToolError(f"Add failed: {str(e)}")
    
    @mcp.tool()
    async def git_push_feature(request: GitPushFeatureRequest) -> str:
        """Push feature branch to origin."""
        try:
            ops = get_ops()
            current = ops.get_current_branch()
            if current == "main":
                return "ERROR: Cannot push main directly."
            if not current.startswith("feature/"):
                return f"ERROR: Invalid branch '{current}'."
            try:
                ops.push("origin", current, force=request.force, no_verify=request.no_verify)
            except RuntimeError as e:
                if "git-lfs" in str(e):
                    ops.push("origin", current, force=request.force, no_verify=True)
                else:
                    raise
            local_hash = ops.get_commit_hash("HEAD")
            pr_url = f"https://github.com/richfrem/Project_Sanctuary/pull/new/{current}"
            return f"Verified push to {current} (Hash: {local_hash[:8]}).\nLink: {pr_url}"
        except Exception as e:
            raise ToolError(f"Push failed: {str(e)}")
    
    @mcp.tool()
    async def git_start_feature(request: GitStartFeatureRequest) -> str:
        """Start a new feature branch."""
        try:
            req_error = check_requirements()
            if req_error:
                return req_error
            return get_ops().start_feature(request.task_id, request.description)
        except Exception as e:
            raise ToolError(f"Start feature failed: {str(e)}")
    
    @mcp.tool()
    async def git_finish_feature(request: GitFinishFeatureRequest) -> str:
        """Finish feature (cleanup/delete)."""
        try:
            return get_ops().finish_feature(request.branch_name, force=request.force)
        except Exception as e:
            raise ToolError(f"Finish feature failed: {str(e)}")
    
    @mcp.tool()
    async def git_diff(request: GitDiffRequest) -> str:
        """Show changes (diff)."""
        try:
            diff = get_ops().diff(cached=request.cached, file_path=request.file_path)
            return diff if diff else "No changes."
        except Exception as e:
            raise ToolError(f"Diff failed: {str(e)}")
    
    @mcp.tool()
    async def git_log(request: GitLogRequest) -> str:
        """Show commit history."""
        try:
            return get_ops().log(max_count=request.max_count, oneline=request.oneline)
        except Exception as e:
            raise ToolError(f"Log retrieval failed: {str(e)}")
    
    logger.info("Starting FastMCP server (STDIO Mode)")
    mcp.run(transport="stdio")


#============================================
# Main Execution Entry Point (ADR-066 v1.3 Canonical Selector)
#============================================
def run_server():
    MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio").lower()
    
    if MCP_TRANSPORT not in {"stdio", "sse"}:
        logger.error(f"Invalid MCP_TRANSPORT: {MCP_TRANSPORT}. Must be 'stdio' or 'sse'.")
        sys.exit(1)
    
    if MCP_TRANSPORT == "sse":
        port = int(os.getenv("PORT", 8000))
        run_sse_server(port)
    else:
        run_stdio_server()


if __name__ == "__main__":
    run_server()
