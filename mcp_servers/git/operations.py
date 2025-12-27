#============================================
# mcp_servers/git/operations.py
# Purpose: Core Logic for Git Operations.
#          Handles executing git commands, parsing output.
# Role: Business Logic Layer
# Used by: mcp_servers.git.server
#============================================

import subprocess
import os
import shutil
from typing import List, Dict, Any, Optional

from mcp_servers.git.models import GitStatus, BranchInfo, RemoteInfo
from mcp_servers.git.validator import GitValidator, ValidationError

class GitOperations:
    """
    Handles git operations with Protocol 101 v3.0 enforcement via GitValidator.
    """
    
    def __init__(self, repo_path: str = ".", base_dir: Optional[str] = None):
        self.repo_path = os.path.abspath(repo_path)
        self.validator = GitValidator()
        
        # Security: Restrict operations to base_dir if specified
        self.base_dir = os.path.abspath(base_dir) if base_dir else None
        if self.base_dir and not self.repo_path.startswith(self.base_dir):
            raise ValueError(f"Repository path {self.repo_path} is outside base directory {self.base_dir}")

    #============================================
    # Method: _get_robust_env
    # Purpose: Create an environment with a robust PATH for git hooks.
    # Returns: Dictionary of environment variables
    #============================================
    #============================================
    # Method: _get_robust_env
    # Purpose: Create an environment with a robust PATH for git hooks.
    # Returns: Dictionary of environment variables
    #============================================
    def _get_robust_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        current_path = env.get("PATH", "")
        
        lfs_path = shutil.which("git-lfs")
        candidates = []
        if lfs_path: candidates.append(os.path.dirname(lfs_path))
        candidates.append(os.path.join(self.repo_path, ".venv", "bin"))
        candidates.append(os.path.join(os.getcwd(), ".venv", "bin"))
        candidates.extend(["/usr/local/bin", "/opt/homebrew/bin", "/usr/bin", "/bin"])
        
        for path in reversed(candidates):
             if path and os.path.isdir(path) and path not in current_path:
                 current_path = f"{path}{os.pathsep}{current_path}"
                 
        env["PATH"] = current_path
        return env

    #============================================
    # Method: _run_git
    # Purpose: Run a git command and return output.
    # Args:
    #   args: List of command arguments
    # Returns: Stdout string
    # Raises: RuntimeError on failure
    #============================================
    #============================================
    # Method: _run_git
    # Purpose: Run a git command and return output.
    # Args:
    #   args: List of command arguments
    # Returns: Stdout string
    # Raises: RuntimeError on failure
    #============================================
    def _run_git(self, args: List[str]) -> str:
        try:
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
            raise RuntimeError(f"Git command failed: {e.stderr.strip()}")

    #============================================
    # Method: get_staged_files
    # Purpose: Get list of currently staged files.
    # Returns: List of filenames
    #============================================
    def get_staged_files(self) -> List[str]:
        output = self._run_git(["diff", "--name-only", "--cached"])
        return output.splitlines() if output else []

    #============================================
    # Method: add
    # Purpose: Stage files for commit.
    # Args:
    #   files: List of files to stage (None for all)
    #   allow_main: Check bypass flag
    #============================================
    #============================================
    # Method: add
    # Purpose: Stage files for commit.
    # Args:
    #   files: List of files to stage (None for all)
    #   allow_main: Check bypass flag
    #============================================
    def add(self, files: List[str] = None, allow_main: bool = False) -> None:
        self.validator.validate_feature_branch_context(self.get_current_branch(), "add", allow_main)
        
        if files is None or len(files) == 0:
            self._run_git(["add", "-A"])
        else:
            self._run_git(["add"] + files)

    #============================================
    # Method: commit
    # Purpose: Commit staged files with enforcement.
    # Args:
    #   message: Commit message
    #   allow_main: Check bypass flag
    # Returns: Commit hash
    #============================================
    #============================================
    # Method: commit
    # Purpose: Commit staged files with enforcement.
    # Args:
    #   message: Commit message
    #   allow_main: Check bypass flag
    # Returns: Commit hash
    #============================================
    def commit(self, message: str, allow_main: bool = False) -> str:
        """Commit staged files with Protocol 101/122 enforcement."""
        self.validator.validate_feature_branch_context(self.get_current_branch(), "commit", allow_main)
        
        # POKA-YOKE via Validator
        staged_files = self.get_staged_files()
        self.validator.validate_poka_yoke(staged_files, lambda f: self.diff(cached=True, file_path=f))
        
        self._run_git(["commit", "-m", message])
        return self._run_git(["rev-parse", "HEAD"])

    #============================================
    # Method: get_current_branch
    # Purpose: Get the current active branch name.
    # Returns: Branch name
    #============================================
    def get_current_branch(self) -> str:
        return self._run_git(["rev-parse", "--abbrev-ref", "HEAD"])

    #============================================
    # Method: get_commit_hash
    # Purpose: Get the full commit hash for a reference.
    # Args:
    #   ref: Git reference (default HEAD)
    # Returns: Commit hash
    #============================================
    def get_commit_hash(self, ref: str = "HEAD") -> str:
        return self._run_git(["rev-parse", ref])

    #============================================
    # Method: create_branch
    # Purpose: Create a new branch.
    # Args:
    #   branch_name: Name of new branch
    #   start_point: Source ref
    #============================================
    def create_branch(self, branch_name: str, start_point: str = "HEAD") -> None:
        self._run_git(["branch", branch_name, start_point])

    #============================================
    # Method: checkout
    # Purpose: Checkout a branch.
    # Args:
    #   branch_name: Branch to checkout
    #============================================
    def checkout(self, branch_name: str) -> None:
        self._run_git(["checkout", branch_name])

    #============================================
    # Method: start_feature
    # Purpose: Start a new feature branch (idempotent).
    # Args:
    #   task_id: Task identifier
    #   description: Short description
    # Returns: Status message
    #============================================
    #============================================
    # Method: start_feature
    # Purpose: Start a new feature branch (idempotent).
    # Args:
    #   task_id: Task identifier
    #   description: Short description
    # Returns: Status message
    #============================================
    def start_feature(self, task_id: str, description: str) -> str:
        status = self.status()
        
        safe_desc = description.lower().replace(" ", "-")
        branch_name = f"feature/task-{task_id}-{safe_desc}"
        
        branch_exists = any(b.name == branch_name for b in status.local_branches)
        
        if branch_exists:
            if status.branch == branch_name:
                return f"Already on feature branch: {branch_name}"
            else:
                self.checkout(branch_name)
                return f"Switched to existing feature branch: {branch_name}"
        else:
            # Validator Logic
            self.validator.validate_one_feature_rule(branch_name, status.feature_branches)
            self.validator.validate_clean_state(status)
            
            self.create_branch(branch_name)
            self.checkout(branch_name)
            return f"Created and switched to new feature branch: {branch_name}"

    #============================================
    # Method: push
    # Purpose: Push to remote.
    # Args:
    #   remote: Remote name
    #   branch: Branch name
    #   force: Force push flag
    #   no_verify: Hook bypass flag
    #   allow_main: Check bypass flag
    # Returns: Git output
    #============================================
    def push(self, remote: str = "origin", branch: str = None, force: bool = False, no_verify: bool = False, allow_main: bool = False) -> str:
        if branch is None:
            branch = self.get_current_branch()
            
        self.validator.validate_feature_branch_context(branch, "push", allow_main)
        
        args = ["push", remote, branch]
        if force: args.append("--force")
        if no_verify: args.append("--no-verify")
            
        return self._run_git(args)

    #============================================
    # Method: pull
    # Purpose: Pull from remote.
    # Args:
    #   remote: Remote name
    #   branch: Branch name
    # Returns: Git output
    #============================================
    def pull(self, remote: str = "origin", branch: str = None) -> str:
        if branch is None:
            branch = self.get_current_branch()
        return self._run_git(["pull", remote, branch])

    #============================================
    # Method: delete_local_branch
    # Purpose: Delete a local branch.
    # Args:
    #   branch_name: Branch to delete
    #   force: Force delete flag
    #============================================
    def delete_local_branch(self, branch_name: str, force: bool = False) -> None:
        flag = "-D" if force else "-d"
        self._run_git(["branch", flag, branch_name])

    #============================================
    # Method: delete_remote_branch
    # Purpose: Delete a remote branch.
    # Args:
    #   branch_name: Branch to delete
    #============================================
    def delete_remote_branch(self, branch_name: str) -> None:
        self._run_git(["push", "origin", "--delete", branch_name])

    #============================================
    # Method: diff_branches
    # Purpose: Get diff between two branches.
    # Args:
    #   branch1: First branch
    #   branch2: Second branch
    # Returns: Diff string
    #============================================
    def diff_branches(self, branch1: str, branch2: str) -> str:
        return self._run_git(["diff", f"{branch1}..{branch2}"])

    #============================================
    # Method: status
    # Purpose: Get comprehensive repo status.
    # Returns: GitStatus model
    #============================================
    #============================================
    # Method: status
    # Purpose: Get comprehensive repo status.
    # Returns: GitStatus model
    #============================================
    def status(self) -> GitStatus:
        current_branch = self.get_current_branch()
        status_porcelain = self._run_git(["status", "--porcelain"])
        
        staged, modified, untracked = [], [], []
        
        for line in status_porcelain.splitlines():
            code = line[:2]
            path = line[2:].strip()
            if code.startswith("M") or code.startswith("A"): staged.append(path)
            if code.endswith("M"): modified.append(path)
            if code.startswith("??"): untracked.append(path)
        
        branches_output = self._run_git(["branch", "-vv"])
        local_branches = []
        for line in branches_output.splitlines():
            is_current = line.startswith("*")
            branch_info = line[2:].strip().split()[0]
            local_branches.append(BranchInfo(name=branch_info, current=is_current))
        
        remote_info = RemoteInfo()
        try:
            upstream = self._run_git(["rev-parse", "--abbrev-ref", f"{current_branch}@{{upstream}}"])
            ahead_behind = self._run_git(["rev-list", "--left-right", "--count", f"{current_branch}...{upstream.strip()}"])
            ahead, behind = ahead_behind.strip().split()
            remote_info = RemoteInfo(upstream=upstream.strip(), ahead=int(ahead), behind=int(behind))
        except RuntimeError:
            pass
        
        feature_branches = [b.name for b in local_branches if b.name.startswith("feature/")]
        is_clean = len(staged) == 0 and len(modified) == 0 and len(untracked) == 0
        
        return GitStatus(
            branch=current_branch,
            staged=staged,
            modified=modified,
            untracked=untracked,
            local_branches=local_branches,
            feature_branches=feature_branches,
            remote=remote_info,
            is_clean=is_clean
        )

    #============================================
    # Method: diff
    # Purpose: Get diff output.
    # Args:
    #   cached: Staged files only
    #   file_path: Specific file
    # Returns: Diff string
    #============================================
    def diff(self, cached: bool = False, file_path: Optional[str] = None) -> str:
        args = ["diff"]
        if cached: args.append("--cached")
        if file_path: args.append(file_path)
        return self._run_git(args)

    #============================================
    # Method: log
    # Purpose: Get commit log.
    # Args:
    #   max_count: Number of commits
    #   oneline: Format flag
    # Returns: Log string
    #============================================
    def log(self, max_count: int = 10, oneline: bool = False) -> str:
        args = ["log", f"-n{max_count}"]
        if oneline: args.append("--oneline")
        return self._run_git(args)

    #============================================
    # Method: finish_feature
    # Purpose: Finish a feature branch (cleanup).
    # Args:
    #   branch_name: Branch to finish
    #   force: Force execution
    # Returns: Status message
    #============================================
    #============================================
    # Method: finish_feature
    # Purpose: Finish a feature branch (cleanup).
    # Args:
    #   branch_name: Branch to finish
    #   force: Force execution
    # Returns: Status message
    #============================================
    def finish_feature(self, branch_name: str, force: bool = False) -> str:
        self.validator.validate_feature_branch_context(branch_name, "finish_feature")
        self.validator.validate_clean_state(self.status())

        # POKA-YOKE: Fetch origin/main first (ADR 074)
        try:
            self._run_git(["fetch", "origin", "main"])
        except RuntimeError as e:
            print(f"WARNING: 'git fetch origin main' failed: {e}")
            print(" proceeding with potentially stale local state.")
        
        if not force:
            feature_commit = self.get_commit_hash(branch_name)
            try:
                self._run_git(["merge-base", "--is-ancestor", feature_commit, "origin/main"])
            except RuntimeError:
                # Fallback: check content match
                self.checkout("main")
                try: self.pull("origin", "main")
                except RuntimeError: pass
                self.checkout(branch_name)
                
                diff_output = self.diff_branches(branch_name, "main")
                if not diff_output or not diff_output.strip():
                     print(f"[POKA-YOKE] Auto-detected squash merge for {branch_name}")
                else:
                    raise RuntimeError(
                        f"[POKA-YOKE] Branch '{branch_name}' is NOT merged into origin/main.\n"
                        "Possible reasons: PR not merged, fetch failed.\n"
                        "Action: Merge PR or use force=True."
                    )
        
        # 4. Delete feature branch
        # Ensure we are off the branch we are deleting
        current_branch = self.get_current_branch()
        if current_branch == branch_name:
            self.checkout("main")
            
        self.delete_local_branch(branch_name, force=True)
            
        return f"Finished feature {branch_name}\nVerified merge (or force bypass)."
