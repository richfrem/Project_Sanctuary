from fastmcp import FastMCP
from mcp_servers.lib.git.git_ops import GitOperations
import os
from typing import List

# Initialize FastMCP with canonical domain name
mcp = FastMCP("project_sanctuary.system.git_workflow")

# Initialize GitOperations
REPO_PATH = os.environ.get("REPO_PATH", ".")
BASE_DIR = os.environ.get("GIT_BASE_DIR", None)
git_ops = GitOperations(REPO_PATH, base_dir=BASE_DIR)

@mcp.tool()
def git_smart_commit(message: str) -> str:
    """
    Commit staged files with automatic Protocol 101 manifest generation.
    
    WORKFLOW: Before calling this tool:
    1. Use git_get_status to see what files have changed
    2. Stage files using standard git commands (git add <files>)
    3. Then call this tool to commit with automatic P101 compliance
    
    Args:
        message: The commit message.
        
    Returns:
        The commit hash or error message.
    """
    try:
        commit_hash = git_ops.commit(message)
        return f"Commit successful. Hash: {commit_hash}"
    except Exception as e:
        return f"Commit failed: {str(e)}"

@mcp.tool()
def git_get_status() -> str:
    """
    Get the current status of the repository.
    
    Returns:
        A formatted string describing the repo status (branch, staged files, etc).
    """
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
def git_add(files: List[str] = None) -> str:
    """
    Stage files for commit.
    
    Args:
        files: List of file paths to stage. If None or empty, stages all changes (git add -A).
        
    Returns:
        Success message.
        
    Example:
        git_add(["core/git/git_ops.py", "tests/test_git_ops.py"])
        git_add()  # Stage all changes
    """
    try:
        git_ops.add(files)
        if files:
            return f"Staged {len(files)} file(s): {', '.join(files)}"
        else:
            return "Staged all changes (git add -A)"
    except Exception as e:
        return f"Failed to stage files: {str(e)}"

@mcp.tool()
def git_push_feature(force: bool = False, no_verify: bool = False) -> str:
    """
    Push the current feature branch to origin.
    
    Args:
        force: Force push (git push --force). Use with caution.
        no_verify: Bypass pre-push hooks (git push --no-verify). Useful if git-lfs is missing.
    
    Returns:
        Push status.
    """
    try:
        current = git_ops.get_current_branch()
        if current == "main":
            return "Error: Cannot push main directly via this tool."
            
        output = git_ops.push("origin", current, force=force, no_verify=no_verify)
        pr_url = f"https://github.com/richfrem/Project_Sanctuary/pull/new/{current}"
        return f"Pushed {current} to origin: {output}\n\n📝 Next: Create PR at {pr_url}"
    except Exception as e:
        return f"Failed to push feature: {str(e)}"

@mcp.tool()
def git_start_feature(task_id: str, description: str) -> str:
    """
    Start a new feature branch.
    Format: feature/task-{task_id}-{description}
    
    Args:
        task_id: The task ID (e.g., "045").
        description: Short description (e.g., "smart-git-mcp").
        
    Returns:
        Success message with branch name.
    """
    try:
        # Sanitize description
        safe_desc = description.lower().replace(" ", "-")
        branch_name = f"feature/task-{task_id}-{safe_desc}"
        
        git_ops.create_branch(branch_name)
        git_ops.checkout(branch_name)
        
        return f"Started feature: {branch_name}"
    except Exception as e:
        return f"Failed to start feature: {str(e)}"

@mcp.tool()
def git_finish_feature(branch_name: str) -> str:
    """
    Finish a feature branch (cleanup).
    Assumes the PR has been merged on GitHub.
    1. Checkout main
    2. Pull latest main
    3. Delete local feature branch
    
    Args:
        branch_name: The branch to finish.
        
    Returns:
        Cleanup status.
    """
    try:
        current = git_ops.get_current_branch()
        if current == branch_name:
            git_ops.checkout("main")
            
        git_ops.pull("origin", "main")
        git_ops.delete_branch(branch_name)
        
        # Delete remote branch
        try:
            git_ops.push("origin", f":{branch_name}")  # Push empty ref to delete remote
        except Exception:
            # Remote branch might already be deleted, that's okay
            pass
        
        return f"Finished feature {branch_name}. Deleted local and remote branches, pulled latest main."
    except Exception as e:
        return f"Failed to finish feature: {str(e)}"

@mcp.tool()
def git_sync_main() -> str:
    """
    Sync the main branch with remote.
    
    Returns:
        Sync status.
    """
    try:
        current = git_ops.get_current_branch()
        if current != "main":
            return "Error: Must be on main branch to sync."
            
        output = git_ops.pull("origin", "main")
        return f"Synced main: {output}"
    except Exception as e:
        return f"Failed to sync main: {str(e)}"

@mcp.tool()
def git_diff(cached: bool = False, file_path: str = None) -> str:
    """
    Show changes in the working directory or staged files.
    
    Args:
        cached: If True, show staged changes. If False, show unstaged changes.
        file_path: Optional specific file to diff.
        
    Returns:
        Diff output.
    """
    try:
        diff_output = git_ops.diff(cached=cached, file_path=file_path)
        if not diff_output:
            return "No changes to display."
        return diff_output
    except Exception as e:
        return f"Failed to get diff: {str(e)}"

@mcp.tool()
def git_log(max_count: int = 10, oneline: bool = False) -> str:
    """
    Show commit history.
    
    Args:
        max_count: Maximum number of commits to show (default: 10).
        oneline: If True, show compact one-line format.
        
    Returns:
        Commit log.
    """
    try:
        return git_ops.log(max_count=max_count, oneline=oneline)
    except Exception as e:
        return f"Failed to get log: {str(e)}"


if __name__ == "__main__":
    mcp.run()
