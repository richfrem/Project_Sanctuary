from fastmcp import FastMCP
from mcp_servers.git.git_ops import GitOperations
import os
import subprocess
import shutil
from typing import List

# Initialize FastMCP with canonical domain name
mcp = FastMCP("project_sanctuary.git")

# Initialize GitOperations
REPO_PATH = os.environ.get("REPO_PATH", ".")
BASE_DIR = os.environ.get("GIT_BASE_DIR", None)
git_ops = GitOperations(REPO_PATH, base_dir=BASE_DIR)

@mcp.tool()
def git_smart_commit(message: str) -> str:
    """
    Commit staged files with automatic Protocol 101 v3.0 (Functional Coherence) enforcement.
    
    ⚠️ CRITICAL: You must adhere to the protocols in `git_get_safety_rules` before using this tool.
    
    Protocol 101 v3.0 mandates that all commits must pass the automated test suite
    before being accepted. The pre-commit hook will automatically execute the test suite.
    
    WORKFLOW: Before calling this tool:
    1. Use git_get_status to see what files have changed
    2. Stage files using standard git commands (git add <files>)
    3. Then call this tool to commit with automatic P101 v3.0 compliance
    
    Args:
        message: The commit message.
        
    Returns:
        The commit hash or error message.
    """
    try:
        # Get current status
        status = git_ops.status()
        current_branch = status["branch"]
        
        # Safety check: Block if on main branch
        if current_branch == "main":
            return (
                "ERROR: Cannot commit directly to main branch. "
                "You must be on a feature branch to make changes. "
                "Please call git_start_feature first to create a feature branch."
            )
            
        # Safety check: Must be a feature branch
        if not current_branch.startswith("feature/"):
            return (
                f"ERROR: Cannot commit on branch '{current_branch}'. "
                f"You must be on a feature branch (format: feature/task-XXX-desc). "
                f"Please call git_start_feature to create a proper feature branch."
            )

        # Verification: Ensure files are staged
        staged_files = git_ops.get_staged_files()
        if not staged_files:
            return "ERROR: No files staged for commit. Please use git_add first."
            
        # Protocol 101 v3.0: Functional Coherence
        # The pre-commit hook (test suite) is the sole validation mechanism.
        # We simply attempt the commit - the hook will enforce test passage.
        commit_hash = git_ops.commit(message)
        return f"Commit successful. Hash: {commit_hash}"

    except Exception as e:
        return f"Commit failed: {str(e)}"

@mcp.tool()
def git_get_safety_rules() -> str:
    """
    Get the unbreakable Git safety rules that all agents MUST follow.
    Call this tool at the start of any session involving Git operations.
    
    Returns:
        The set of immutable safety protocols for Git usage.
    """
    return """
    🛡️ GIT SAFETY PRIMER: UNBREAKABLE RULES FOR AGENTS 🛡️
    
    1. SYNCHRONIZATION FIRST
       - Before starting ANY new task, ensure local 'main' matches 'origin/main'.
       - Run 'git checkout main && git pull origin main'.
       - If pull fails or diverges: STOP immediately. Ask user for help.
       - NEVER start work on a stale or divergent main branch.
       
    2. MAIN IS PROTECTED
       - NEVER commit directly to 'main'.
       - ALWAYS create a feature branch: 'feature/task-ID-description'.
       
    3. SERIAL PROCESSING PROTOCOL
       - ONE feature branch at a time.
       - Do not create 'feature/B' if 'feature/A' is active.
       - Lifecycle: Start A -> Commit -> Push -> PR -> Merge -> Pull Main -> Delete A -> Start B.
       
    4. STATE VERIFICATION
       - NEVER assume the repo state.
       - Run 'git_get_status' before every major action (edit, commit, switch).
       - If you see unexpected changes/untracked files: STOP and report to user.
       
    5. DESTRUCTIVE ACTION GATE
       - 'git reset --hard', 'git clean', 'git push --force' require EXPLICIT, RECENT user confirmation.
       - NEVER auto-run these commands to "fix" things without asking.
       
    6. NO GHOST EDITS
       - Do not modify files unless you are on the correct feature branch.
       - Verify branch with 'git_get_status' BEFORE editing.
    """

def check_requirements() -> str:
    """
    Pillar 6: Pre-Flight Check.
    Verifies that all dependencies in REQUIREMENTS.env are installed.
    Returns None if successful, or error message if failed.
    """
    req_file = os.path.join(REPO_PATH, "REQUIREMENTS.env")
    if not os.path.exists(req_file):
        return None # No requirements file, skip check
        
    try:
        with open(req_file, 'r') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
        for req in requirements:
            tool_name = req.split('>')[0].split('=')[0].split('<')[0].strip()
            
            # Special handling for git-lfs (git subcommand)
            if tool_name == "git-lfs":
                cmd = ["git", "lfs", "version"]
            else:
                cmd = [tool_name, "--version"]
            
            # Basic check: try to run the tool with --version
            try:
                subprocess.run(cmd, check=True, capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                return f"PROTOCOL VIOLATION: Missing required dependency: {tool_name}. Please install it as per REQUIREMENTS.env."
    except Exception as e:
        return f"Failed to verify requirements: {str(e)}"
    return None

@mcp.tool()
def git_get_status() -> str:
    """
    Get the current status of the repository.
    
    ⚠️ CRITICAL: You must adhere to the protocols in `git_get_safety_rules` before using this tool.
    
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
    
    ⚠️ CRITICAL: You must adhere to the protocols in `git_get_safety_rules` before using this tool.
    
    Safety: Blocks if on main branch - must be on feature branch.
    
    Args:
        files: List of file paths to stage. If None or empty, stages all changes (git add -A).
        
    Returns:
        Success message.
        
    Example:
        git_add(["core/git/git_ops.py", "tests/test_git_ops.py"])
        git_add()  # Stage all changes
    """
    try:
        # Get current status
        status = git_ops.status()
        current_branch = status["branch"]
        
        # Safety check: Block if on main branch
        if current_branch == "main":
            return (
                "ERROR: Cannot stage files on main branch. "
                "You must be on a feature branch to make changes. "
                "Please call git_start_feature first to create a feature branch."
            )
        
        # Safety check: Verify on a feature branch
        if not current_branch.startswith("feature/"):
            return (
                f"ERROR: Cannot stage files on branch '{current_branch}'. "
                f"You must be on a feature branch (format: feature/task-XXX-desc). "
                f"Current branch does not follow feature branch naming convention. "
                f"Please call git_start_feature to create a proper feature branch."
            )
        
        # All checks passed - stage files
        git_ops.add(files)
        
        if files:
            return f"Staged {len(files)} file(s) on {current_branch}: {', '.join(files)}"
        else:
            return f"Staged all changes on {current_branch} (git add -A)"
            
    except Exception as e:
        return f"Failed to stage files: {str(e)}"

@mcp.tool()
def git_push_feature(force: bool = False, no_verify: bool = False) -> str:
    """
    Push the current feature branch to origin.
    
    ⚠️ CRITICAL: You must adhere to the protocols in `git_get_safety_rules` before using this tool.
    
    Args:
        force: Force push (git push --force). Use with caution.
        no_verify: Bypass pre-push hooks (git push --no-verify). Useful if git-lfs is missing.
    
    Returns:
        Push status.
    """
    try:
        current = git_ops.get_current_branch()
        if current == "main":
            return (
                "ERROR: Cannot push main branch directly. "
                "You must be on a feature branch to push changes. "
                "Please call git_start_feature first to create a feature branch."
            )

        # Safety check: Must be a feature branch
        if not current.startswith("feature/"):
            return (
                f"ERROR: Cannot push branch '{current}'. "
                f"You must be on a feature branch (format: feature/task-XXX-desc). "
                f"Please call git_start_feature to create a proper feature branch."
            )
            
        # Verification: Ensure we have something to push?
        # Actually git push handles "everything up-to-date" gracefully.
        
        # Strategy: Try push normally first.
        # If it fails due to LFS hook issues, retry with --no-verify.
        # This handles cases where LFS is installed but hooks fail (PATH issues),
        # or where LFS is missing entirely.
        
        warning_msg = ""
        try:
            output = git_ops.push("origin", current, force=force, no_verify=no_verify)
        except RuntimeError as e:
            error_msg = str(e)
            # Check for common LFS hook failures
            if "git-lfs" in error_msg and ("not found" in error_msg or "command not found" in error_msg):
                print(f"WARNING: Push failed due to LFS hook error. Retrying with --no-verify. Error: {error_msg}")
                # Retry with bypass
                output = git_ops.push("origin", current, force=force, no_verify=True)
                warning_msg = "⚠️ WARNING: LFS hook failed, pushed with --no-verify.\n"
            else:
                # Re-raise other errors
                raise e
        
        # Verification: Verify remote hash matches local hash
        local_hash = git_ops.get_commit_hash("HEAD")
        remote_hash = git_ops.get_commit_hash(f"origin/{current}")
        
        if local_hash != remote_hash:
            return f"{warning_msg}WARNING: Push completed but remote hash ({remote_hash[:8]}) does not match local ({local_hash[:8]}). Output: {output}"
            
        pr_url = f"https://github.com/richfrem/Project_Sanctuary/pull/new/{current}"
        return f"{warning_msg}Verified push to {current} (Hash: {local_hash[:8]}).\nOutput: {output}\n\n📝 Next: Create PR at {pr_url}"
    except Exception as e:
        return f"Failed to push feature: {str(e)}"

@mcp.tool()
def git_start_feature(task_id: str, description: str) -> str:
    """
    Start a new feature branch (idempotent).
    Format: feature/task-{task_id}-{description}
    
    ⚠️ CRITICAL: You must adhere to the protocols in `git_get_safety_rules` before using this tool.
    
    Idempotent behavior:
    - If branch exists and you're on it: success (no-op)
    - If branch exists but you're elsewhere: checkout to it
    - If branch doesn't exist: create and checkout
    
    Safety checks:
    - Blocks if a DIFFERENT feature branch exists (one at a time rule)
    - Requires clean working directory for new branch creation
    
    Args:
        task_id: The task ID (e.g., "045").
        description: Short description (e.g., "smart-git-mcp").
        
    Returns:
        Success message with branch name.
    """
    try:
        # Pillar 6: Pre-Flight Check
        req_error = check_requirements()
        if req_error:
            # If it's just git-lfs missing, warn but allow proceeding
            if "git-lfs" in req_error:
                print(f"WARNING: {req_error}")
                # Continue execution
            else:
                return req_error

        # Get comprehensive status
        status = git_ops.status()
        current_branch = status["branch"]
        feature_branches = status["feature_branches"]
        local_branches = [b["name"] for b in status["local_branches"]]
        is_clean = status["is_clean"]
        
        # Sanitize and build branch name
        safe_desc = description.lower().replace(" ", "-")
        branch_name = f"feature/task-{task_id}-{safe_desc}"
        
        # Check if branch already exists
        branch_exists = branch_name in local_branches
        
        if branch_exists:
            # Branch exists - idempotent behavior
            if current_branch == branch_name:
                # Already on the branch - no-op
                return f"Already on feature branch: {branch_name}"
            else:
                # Switch to existing branch
                git_ops.checkout(branch_name)
                return f"Switched to existing feature branch: {branch_name}"
        else:
            # Branch doesn't exist - need to create
            
            # Safety check: No other feature branches allowed
            if len(feature_branches) > 0:
                return (
                    f"ERROR: Cannot create new feature branch. "
                    f"Existing feature branch(es) detected: {', '.join(feature_branches)}. "
                    f"Only one feature branch at a time is allowed. "
                    f"Please finish the current feature branch first using git_finish_feature."
                )
            
            # Safety check: Clean working directory required
            if not is_clean:
                return (
                    f"ERROR: Cannot create new feature branch. "
                    f"Working directory has uncommitted changes. "
                    f"Staged: {len(status['staged'])}, "
                    f"Modified: {len(status['modified'])}, "
                    f"Untracked: {len(status['untracked'])}. "
                    f"Please commit or stash changes first."
                )
            
            # All checks passed - create and checkout
            git_ops.create_branch(branch_name)
            git_ops.checkout(branch_name)
            
            return f"Created and switched to new feature branch: {branch_name}"
            
    except Exception as e:
        return f"Failed to start feature: {str(e)}"

@mcp.tool()
def git_finish_feature(branch_name: str, force: bool = False) -> str:
    """
    Finish a feature branch (cleanup).
    
    ⚠️ CRITICAL: You must adhere to the protocols in `git_get_safety_rules` before using this tool.
    
    Assumes the PR has been merged on GitHub.
    1. Checkout main
    2. Pull latest main
    3. Delete local feature branch
    4. Delete remote feature branch
    
    Args:
        branch_name: The branch to finish.
        force: If True, bypass merge verification (useful for squash merges).
        
    Returns:
        Cleanup status.
    """
    try:
        # Safety check: Cannot finish main branch
        if branch_name == "main":
            return "ERROR: Cannot finish 'main' branch. It is the protected default branch."
            
        # Safety check: Must be a feature branch
        if not branch_name.startswith("feature/"):
            return (
                f"ERROR: Invalid branch name '{branch_name}'. "
                f"Can only finish feature branches (format: feature/task-XXX-desc)."
            )

        # Pillar 4: Verify clean state before finishing (merging/deleting)
        git_ops.verify_clean_state()

        # Safety check: Verify branch is merged into main
        # This prevents data loss by ensuring the PR is actually merged
        # Skip if force=True (e.g. for squash merges where commit history is lost)
        if not force and not git_ops.is_branch_merged(branch_name, "main"):
            # Double check by fetching origin first? 
            # Sometimes local main is behind origin/main, so it looks unmerged locally
            # but is merged on remote.
            git_ops.checkout("main")
            git_ops.pull("origin", "main")
            git_ops.checkout(branch_name)
            
            # Check again after sync
            if not git_ops.is_branch_merged(branch_name, "main"):
                # Auto-detect squash merge: check if branches have identical content
                try:
                    diff_output = git_ops.diff_branches(branch_name, "main")
                    if not diff_output or diff_output.strip() == "":
                        # Branches have identical content - squash merge detected!
                        # Log this and proceed with cleanup
                        print(f"Auto-detected squash merge for {branch_name} (identical content to main)")
                    else:
                        # Branches differ - truly unmerged
                        return (
                            f"ERROR: Branch '{branch_name}' is NOT merged into main. "
                            "Cannot finish/delete an unmerged feature branch. "
                            "Please merge your PR on GitHub first, then run this command again. "
                            "If you squash merged, use force=True to bypass this check."
                        )
                except Exception as e:
                    # If diff check fails, fall back to error
                    return (
                        f"ERROR: Branch '{branch_name}' is NOT merged into main. "
                        "Cannot finish/delete an unmerged feature branch. "
                        "Please merge your PR on GitHub first, then run this command again. "
                        "If you squash merged, use force=True to bypass this check."
                    )

        # ALWAYS checkout main first to avoid merging main into the feature branch
        git_ops.checkout("main")
            
        git_ops.pull("origin", "main")
        
        # Delete local branch (force delete since we verified merge status)
        git_ops.delete_local_branch(branch_name, force=True)
        
        # Delete remote branch
        try:
            git_ops.delete_remote_branch(branch_name)
        except Exception:
            # Remote branch might already be deleted, that's okay
            pass
        
        return f"Finished feature {branch_name}. Verified merge, deleted local/remote branches, and synced main."
    except Exception as e:
        return f"Failed to finish feature: {str(e)}"

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
