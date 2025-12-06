import subprocess
import os
from typing import List, Dict, Any, Optional

class GitOperations:
    """
    Handles git operations with Protocol 101 v3.0 (Functional Coherence) enforcement.
    
    Protocol 101 v3.0 mandates that all commits must pass the automated test suite
    before being accepted. This class provides safe, whitelisted git operations.
    """
    
    def __init__(self, repo_path: str = ".", base_dir: Optional[str] = None):
        self.repo_path = os.path.abspath(repo_path)
        
        # Security: Restrict operations to base_dir if specified
        self.base_dir = os.path.abspath(base_dir) if base_dir else None
        if self.base_dir and not self.repo_path.startswith(self.base_dir):
            raise ValueError(f"Repository path {self.repo_path} is outside base directory {self.base_dir}")

    def verify_clean_state(self) -> None:
        """
        Pillar 4: Pre-Execution Verification.
        Ensures the working directory is clean before critical operations.
        Raises RuntimeError if dirty.
        """
        status = self.status()
        if status["modified"] or status["staged"] or status["untracked"]:
            raise RuntimeError(
                f"Working directory is not clean. "
                f"Modified: {len(status['modified'])}, "
                f"Staged: {len(status['staged'])}, "
                f"Untracked: {len(status['untracked'])}. "
                "Please commit or stash changes before proceeding."
            )

    def _get_robust_env(self) -> Dict[str, str]:
        """
        Create an environment with a robust PATH for git hooks.
        Ensures tools like git-lfs are visible to hooks (e.g. post-checkout).
        """
        import shutil
        env = os.environ.copy()
        current_path = env.get("PATH", "")
        
        # 1. Dynamically locate git-lfs
        lfs_path = shutil.which("git-lfs")
        candidates = []
        
        if lfs_path:
            candidates.append(os.path.dirname(lfs_path))
            
        # 2. Add standard system locations (portability fallback)
        standard_paths = [
            "/usr/local/bin",      # Intel Mac / Linux
            "/opt/homebrew/bin",   # Apple Silicon Mac
            "/usr/bin",            # Standard
            "/bin"                 # Standard
        ]
        candidates.extend(standard_paths)
        
        # 3. Prepend valid paths to PATH
        for path in candidates:
             if path and os.path.isdir(path) and path not in current_path:
                 current_path = f"{path}{os.pathsep}{current_path}"
                 
        env["PATH"] = current_path
        
        # Debug logging to identify detection issues
        import sys
        print(f"[DEBUG] git_ops detected PATH for hooks: {current_path}", file=sys.stderr)
        
        return env

    def _run_git(self, args: List[str]) -> str:
        """Run a git command and return output."""
        try:
            # Use robust environment
            env = self._get_robust_env()

            result = subprocess.run(
                ["git"] + args,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True,
                env=env
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            # Enhanced error handling to capture stderr
            raise RuntimeError(f"Git command failed: {e.stderr.strip()}")

    def get_staged_files(self) -> List[str]:
        """Get list of currently staged files."""
        output = self._run_git(["diff", "--name-only", "--cached"])
        if not output:
            return []
        return output.splitlines()

    def add(self, files: List[str] = None) -> None:
        """Stage files for commit."""
        if files is None or len(files) == 0:
            # Stage all modified and new files
            self._run_git(["add", "-A"])
        else:
            self._run_git(["add"] + files)

    # PROTOCOL 101 v3.0: Manifest generation methods PERMANENTLY REMOVED
    # Functional Coherence (test suite execution) is now the sole integrity mechanism

    def commit(self, message: str) -> str:
        """
        Commit staged files with Protocol 101 v3.0 compliance.
        
        Protocol 101 v3.0 (Functional Coherence):
        - The pre-commit hook will automatically execute ./scripts/run_genome_tests.sh
        - All tests must pass for the commit to proceed
        - No manifest generation is required
        
        Returns commit hash.
        """
        # Protocol 101 v3.0: Pre-commit hook handles test execution
        # We simply commit normally - the hook will enforce functional coherence
        self._run_git(["commit", "-m", message])
        
        # Return hash
        return self._run_git(["rev-parse", "HEAD"])

    def get_current_branch(self) -> str:
        """Get the current active branch name."""
        return self._run_git(["rev-parse", "--abbrev-ref", "HEAD"])

    def get_commit_hash(self, ref: str = "HEAD") -> str:
        """Get the full commit hash for a reference."""
        return self._run_git(["rev-parse", ref])

    def create_branch(self, branch_name: str, start_point: str = "HEAD") -> None:
        """Create a new branch."""
        self._run_git(["branch", branch_name, start_point])

    def checkout(self, branch_name: str) -> None:
        """Checkout a branch."""
        self._run_git(["checkout", branch_name])

    def push(self, remote: str = "origin", branch: str = None, force: bool = False, no_verify: bool = False) -> str:
        """Push to remote."""
        if branch is None:
            branch = self.get_current_branch()
        
        args = ["push", remote, branch]
        if force:
            args.append("--force")
        if no_verify:
            args.append("--no-verify")
            
        return self._run_git(args)

    def pull(self, remote: str = "origin", branch: str = None) -> str:
        """Pull from remote."""
        if branch is None:
            branch = self.get_current_branch()
        return self._run_git(["pull", remote, branch])

    def delete_branch(self, branch_name: str, force: bool = False) -> None:
        """Delete a branch."""
        flag = "-D" if force else "-d"
        self._run_git(["branch", flag, branch_name])

    def delete_local_branch(self, branch_name: str, force: bool = False) -> None:
        """Delete a local branch (alias for delete_branch)."""
        self.delete_branch(branch_name, force)

    def delete_remote_branch(self, branch_name: str) -> None:
        """Delete a remote branch."""
        self._run_git(["push", "origin", "--delete", branch_name])

    def is_branch_merged(self, branch_name: str, target_branch: str = "main") -> bool:
        """Check if a branch is merged into the target branch."""
        try:
            # Get list of branches merged into target
            output = self._run_git(["branch", "--merged", target_branch])
            merged_branches = [b.strip().replace("* ", "") for b in output.splitlines()]
            return branch_name in merged_branches
        except Exception:
            return False

    def status(self) -> Dict[str, Any]:
        """Get comprehensive repo status including branches and remote tracking."""
        current_branch = self.get_current_branch()
        status_porcelain = self._run_git(["status", "--porcelain"])
        
        # Parse file status
        staged = []
        modified = []
        untracked = []
        
        for line in status_porcelain.splitlines():
            code = line[:2]
            path = line[3:]
            if code.startswith("M") or code.startswith("A"):
                staged.append(path)
            if code.endswith("M"):
                modified.append(path)
            if code.startswith("??"):
                untracked.append(path)
        
        # Get all local branches
        branches_output = self._run_git(["branch", "-vv"])
        local_branches = []
        for line in branches_output.splitlines():
            is_current = line.startswith("*")
            branch_info = line[2:].strip()  # Remove "* " or "  "
            local_branches.append({
                "name": branch_info.split()[0],
                "current": is_current
            })
        
        # Get remote tracking info for current branch
        remote_info = {}
        try:
            # Get upstream branch
            upstream = self._run_git(["rev-parse", "--abbrev-ref", f"{current_branch}@{{upstream}}"])
            remote_info["upstream"] = upstream.strip()
            
            # Get ahead/behind counts
            ahead_behind = self._run_git(["rev-list", "--left-right", "--count", f"{current_branch}...{upstream.strip()}"])
            ahead, behind = ahead_behind.strip().split()
            remote_info["ahead"] = int(ahead)
            remote_info["behind"] = int(behind)
        except RuntimeError:
            # No upstream configured
            remote_info["upstream"] = None
            remote_info["ahead"] = 0
            remote_info["behind"] = 0
        
        # Count feature branches (for safety check)
        feature_branches = [b["name"] for b in local_branches if b["name"].startswith("feature/")]
        
        return {
            "branch": current_branch,
            "staged": staged,
            "modified": modified,
            "untracked": untracked,
            "local_branches": local_branches,
            "feature_branches": feature_branches,
            "remote": remote_info,
            "is_clean": len(staged) == 0 and len(modified) == 0 and len(untracked) == 0
        }

    def diff(self, cached: bool = False, file_path: Optional[str] = None) -> str:
        """Get diff output."""
        args = ["diff"]
        if cached:
            args.append("--cached")
        if file_path:
            args.append(file_path)
        return self._run_git(args)

    def log(self, max_count: int = 10, oneline: bool = False) -> str:
        """Get commit log."""
        args = ["log", f"-n{max_count}"]
        if oneline:
            args.append("--oneline")
        return self._run_git(args)

    def diff_branches(self, branch1: str, branch2: str) -> str:
        """
        Get diff between two branches.
        
        Args:
            branch1: First branch name
            branch2: Second branch name
            
        Returns:
            Diff output (empty string if branches have identical content)
        """
        return self._run_git(["diff", f"{branch1}..{branch2}"])




