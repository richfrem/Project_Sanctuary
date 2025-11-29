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

    def _run_git(self, args: List[str]) -> str:
        """Run a git command and return output."""
        try:
            result = subprocess.run(
                ["git"] + args,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
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

    def status(self) -> Dict[str, Any]:
        """Get repo status."""
        branch = self.get_current_branch()
        status_porcelain = self._run_git(["status", "--porcelain"])
        
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
                
        return {
            "branch": branch,
            "staged": staged,
            "modified": modified,
            "untracked": untracked
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




