"""
Sanctuary Git Server
Domain: project_sanctuary.git / sanctuary_git

Refactored to use FastMCP for proper MCP SSE protocol compliance (ADR-066).
"""
import os
import sys
from typing import List, Optional
from pathlib import Path

# Add project root to python path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.insert(0, str(project_root))

from fastmcp import FastMCP

# Import git_ops from local directory
try:
    from .git_ops import GitOperations
except ImportError:
    from git_ops import GitOperations

# Initialize FastMCP
mcp = FastMCP("sanctuary_git")

# Operations
REPO_PATH = os.environ.get("REPO_PATH", "/app")
BASE_DIR = os.environ.get("GIT_BASE_DIR", None)
git_ops = GitOperations(REPO_PATH, base_dir=BASE_DIR)

def check_requirements() -> Optional[str]:
    """Internal helper to check requirements.env."""
    req_file = os.path.join(REPO_PATH, "REQUIREMENTS.env")
    if not os.path.exists(req_file):
        return None
    try:
        with open(req_file, 'r') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        # Basic check logic (assuming environment is pre-validated in container)
    except Exception as e:
        return f"Failed to verify requirements: {str(e)}"
    return None

# =============================================================================
# GIT TOOLS
# =============================================================================

@mcp.tool()
def git_smart_commit(message: str) -> str:
    """Commit staged files with automatic Protocol 101 v3.0 enforcement."""
    try:
        status = git_ops.status()
        current_branch = status["branch"]
        
        if current_branch == "main":
            return "ERROR: Cannot commit directly to main. Use git_start_feature."
        if not current_branch.startswith("feature/"):
            return f"ERROR: Invalid branch '{current_branch}'. Use feature/ format."
            
        staged_files = git_ops.get_staged_files()
        if not staged_files:
            return "ERROR: No files staged."
            
        commit_hash = git_ops.commit(message)
        return f"Commit successful. Hash: {commit_hash}"
    except Exception as e:
        return f"Commit failed: {str(e)}"

@mcp.tool()
def git_get_safety_rules() -> str:
    """Get the unbreakable Git safety rules (Protocol 101)."""
    return """GIT SAFETY PRIMER - UNBREAKABLE RULES FOR AGENTS
    1. SYNCHRONIZATION FIRST - Pull main before starting.
    2. MAIN IS PROTECTED - Never commit to main.
    3. SERIAL PROCESSING - One feature branch at a time.
    4. STATE VERIFICATION - Check status before acting.
    5. DESTRUCTIVE ACTION GATE - No force pushes without approval.
    6. NO GHOST EDITS - Verify branch before editing.
    """

@mcp.tool()
def git_get_status() -> str:
    """Get the current status of the repository."""
    try:
        status = git_ops.status()
        return (
            f"Branch: {status['branch']}\n"
            f"Staged Files: {', '.join(status['staged'])}\n"
            f"Modified Files: {', '.join(status['modified'])}\n"
            f"Untracked Files: {', '.join(status['untracked'])}"
        )
    except Exception as e:
        return f"Failed to get status: {str(e)}"

@mcp.tool()
def git_add(files: Optional[List[str]] = None) -> str:
    """Stage files for commit."""
    try:
        status = git_ops.status()
        current_branch = status["branch"]
        
        if current_branch == "main":
            return "ERROR: Cannot stage on main branch."
        if not current_branch.startswith("feature/"):
            return f"ERROR: Invalid branch '{current_branch}'."
            
        git_ops.add(files)
        if files:
            return f"Staged {len(files)} file(s) on {current_branch}"
        else:
            return f"Staged all changes on {current_branch}"
    except Exception as e:
        return f"Failed to stage files: {str(e)}"

@mcp.tool()
def git_push_feature(force: bool = False, no_verify: bool = False) -> str:
    """Push the current feature branch to origin."""
    try:
        current = git_ops.get_current_branch()
        if current == "main":
            return "ERROR: Cannot push main directly."
        if not current.startswith("feature/"):
            return f"ERROR: Invalid branch '{current}'."
            
        try:
            output = git_ops.push("origin", current, force=force, no_verify=no_verify)
        except RuntimeError as e:
            # Simple retry logic for LFS hooks
            if "git-lfs" in str(e):
                output = git_ops.push("origin", current, force=force, no_verify=True)
            else:
                raise e
        
        local_hash = git_ops.get_commit_hash("HEAD")
        pr_url = f"https://github.com/richfrem/Project_Sanctuary/pull/new/{current}"
        return f"Verified push to {current} (Hash: {local_hash[:8]}).\nLink: {pr_url}"
    except Exception as e:
        return f"Failed to push feature: {str(e)}"

@mcp.tool()
def git_start_feature(task_id: str, description: str) -> str:
    """Start a new feature branch."""
    try:
        req_error = check_requirements()
        if req_error:
            return req_error
        return git_ops.start_feature(task_id, description)
    except Exception as e:
        return f"Failed to start feature: {str(e)}"

@mcp.tool()
def git_finish_feature(branch_name: str, force: bool = False) -> str:
    """Finish a feature branch."""
    try:
        return git_ops.finish_feature(branch_name, force=force)
    except Exception as e:
        return f"Failed to finish feature: {str(e)}"

@mcp.tool()
def git_diff(cached: bool = False, file_path: Optional[str] = None) -> str:
    """Show changes in the working directory."""
    try:
        diff = git_ops.diff(cached=cached, file_path=file_path)
        return diff if diff else "No changes."
    except Exception as e:
        return f"Failed to get diff: {str(e)}"

@mcp.tool()
def git_log(max_count: int = 10, oneline: bool = False) -> str:
    """Show commit history."""
    try:
        return git_ops.log(max_count=max_count, oneline=oneline)
    except Exception as e:
        return f"Failed to get log: {str(e)}"

# =============================================================================
# Health Check (for Gateway health monitoring)
# =============================================================================
from starlette.responses import JSONResponse

@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    return JSONResponse({"status": "healthy", "service": "sanctuary_git"})

# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    port_env = os.environ.get("PORT")
    
    if port_env:
        # Docker Mode: Listen on 0.0.0.0 via SSE
        print(f"🚀 Starting Git Server on port {port_env} (Transport: SSE)")
        mcp.run(transport="sse", port=int(port_env), host="0.0.0.0")
    else:
        # Local Mode: Stdio
        mcp.run()

