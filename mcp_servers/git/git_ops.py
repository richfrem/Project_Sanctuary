import subprocess
import os
from typing import List, Dict, Any, Optional

# Poka-Yoke: High-Risk File List (Protocol 122)
# These files require security audit before committing
HIGH_RISK_FILES = [".gitignore", ".env", ".env.local", "Dockerfile", "package.json"]

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

    def _verify_feature_branch(self, operation: str, allow_main: bool) -> None:
        """Helper to enforce feature usage."""
        if allow_main:
            return
            
        current = self.get_current_branch()
        if current == "main":
            raise ValueError(f"SAFETY ERROR: Cannot perform '{operation}' on 'main' branch. Switch to a feature branch.")
            
    def add(self, files: List[str] = None, allow_main: bool = False) -> None:
        """Stage files for commit."""
        self._verify_feature_branch("add", allow_main)
        
        # User Safety Request: "make sure when doing git add you do git status"
        status = self.status()
        
        if files is None or len(files) == 0:
            # Stage all modified and new files
            self._run_git(["add", "-A"])
        else:
            self._run_git(["add"] + files)

    # PROTOCOL 101 v3.0: Manifest generation methods PERMANENTLY REMOVED
    # Functional Coherence (test suite execution) is now the sole integrity mechanism

    def commit(self, message: str, allow_main: bool = False) -> str:
        """
        Commit staged files with Protocol 101 v3.0 compliance and Poka-Yoke security audit.
        
        Protocol 101 v3.0 (Functional Coherence):
        - The pre-commit hook will automatically execute ./scripts/run_genome_tests.sh
        - All tests must pass for the commit to proceed
        
        Poka-Yoke (Protocol 122):
        - Staged high-risk files trigger a blocking security audit
        - Audit checks for content deletion and secret patterns
        
        Returns commit hash.
        """
        self._verify_feature_branch("commit", allow_main)
        
        # POKA-YOKE: Check staged files for high-risk items
        staged_files = self.get_staged_files()
        high_risk_staged = [f for f in staged_files if any(f.endswith(hr) for hr in HIGH_RISK_FILES)]
        
        if high_risk_staged:
            self._poka_yoke_audit(high_risk_staged)
        
        # Protocol 101 v3.0: Pre-commit hook handles test execution
        self._run_git(["commit", "-m", message])
        
        return self._run_git(["rev-parse", "HEAD"])

    def _poka_yoke_audit(self, high_risk_files: List[str]) -> None:
        """
        Poka-Yoke: Blocking security audit for high-risk files in staging.
        
        Checks for:
        1. Significant content deletion (lines removed > 2x lines added)
        2. Secret patterns in diff
        
        Raises RuntimeError if audit fails.
        """
        import sys
        
        print(f"[POKA-YOKE] High-risk files detected in staging: {high_risk_files}", file=sys.stderr)
        
        for file_path in high_risk_files:
            # Get diff for this file
            diff_output = self.diff(cached=True, file_path=file_path)
            
            # Check for content deletion (lines removed > lines added * 2)
            lines_added = diff_output.count('\n+') - 1  # Subtract header line
            lines_removed = diff_output.count('\n-') - 1
            
            if lines_removed > 0 and lines_removed > lines_added * 2:
                raise RuntimeError(
                    f"POKA-YOKE BLOCKED: Commit rejected. High-risk file '{file_path}' "
                    f"has significant content removal (removed: {lines_removed}, added: {lines_added}). "
                    f"This may indicate accidental clearing. Review the diff carefully."
                )
            
            # Check for secrets (basic pattern matching)
            secret_patterns = ["API_KEY=", "SECRET=", "PASSWORD=", "TOKEN=", "aws_secret"]
            for pattern in secret_patterns:
                if pattern.lower() in diff_output.lower():
                    raise RuntimeError(
                        f"POKA-YOKE BLOCKED: Commit rejected. Potential secret detected in '{file_path}'. "
                        f"Pattern matched: {pattern}. Remove secrets before committing."
                    )
        
        print(f"[POKA-YOKE] Security audit PASSED for: {high_risk_files}", file=sys.stderr)

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

    def start_feature(self, task_id: str, description: str) -> str:
        """
        Start a new feature branch (idempotent).
        
        Logic:
        1. Sanitize name
        2. Check for existing feature branches (One Feature Rule)
        3. Check for clean working dir
        4. Create & Checkout (or switch if exists)
        """
        # Get comprehensive status
        status = self.status()
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
                return f"Already on feature branch: {branch_name}"
            else:
                self.checkout(branch_name)
                return f"Switched to existing feature branch: {branch_name}"
        else:
            # Branch doesn't exist - need to create
            
            # Safety check: No other feature branches allowed (User Rule)
            if len(feature_branches) > 0:
                raise RuntimeError(
                    f"One Feature Rule: Cannot create '{branch_name}'. "
                    f"Existing feature branch(es) detected: {', '.join(feature_branches)}. "
                    f"Please finish the current feature branch first."
                )
            
            # Safety check: Clean working directory required
            if not is_clean:
                raise RuntimeError(
                    f"Cannot create new feature branch. Working directory has uncommitted changes. "
                    f"Please commit or stash changes first."
                )
            
            # All checks passed - create and checkout
            self.create_branch(branch_name)
            self.checkout(branch_name)
            
            return f"Created and switched to new feature branch: {branch_name}"

    def push(self, remote: str = "origin", branch: str = None, force: bool = False, no_verify: bool = False, allow_main: bool = False) -> str:
        """Push to remote."""
        if branch is None:
            branch = self.get_current_branch()
            
        if branch == "main" and not allow_main:
             raise ValueError("SAFETY ERROR: Cannot push 'main' branch directly. Use a PR.")
        
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
            # Use strip() to handle potential variable whitespace separators
            # (e.g., "M  file" vs "M file" if some git versions differ)
            path = line[2:].strip()
            
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

    def finish_feature(self, branch_name: str, force: bool = False) -> str:
        """
        Finish a feature branch (cleanup).
        
        Logic:
        1. Verify clean state
        2. Verify merge status (or squash detection)
        3. Checkout main & pull
        4. Delete local & remote branches
        """
        # Safety check: Cannot finish main branch
        if branch_name == "main":
            raise ValueError("Cannot finish 'main' branch. It is the protected default branch.")
            
        # Safety check: Must be a feature branch
        if not branch_name.startswith("feature/"):
             raise ValueError(f"Invalid branch name '{branch_name}'. Can only finish feature branches.")

        # Pillar 4: Verify clean state before finishing
        self.verify_clean_state()

        # POKA-YOKE: Fetch origin/main first to ensure we have latest remote state
        try:
            self._run_git(["fetch", "origin", "main"])
        except RuntimeError:
            pass  # Fetch may fail if offline, continue with local check
        
        # Safety check: Verify branch is merged into ORIGIN main (the remote source of truth)
        if not force:
            # Get the commit hash of the feature branch
            feature_commit = self.get_commit_hash(branch_name)
            
            # Check if this commit exists in origin/main
            try:
                # This will succeed silently if commit is an ancestor of origin/main
                self._run_git(["merge-base", "--is-ancestor", feature_commit, "origin/main"])
                # If we get here, commit is in origin/main - safe to proceed
            except RuntimeError:
                # Commit not in origin/main - check for squash merge via content comparison
                # First sync local main with origin
                self.checkout("main")
                self.pull("origin", "main")
                self.checkout(branch_name)
                
                # Compare content - if identical, squash merge occurred
                diff_output = self.diff_branches(branch_name, "main")
                if not diff_output or diff_output.strip() == "":
                    # Branches have identical content - squash merge detected
                    print(f"[POKA-YOKE] Auto-detected squash merge for {branch_name} (identical content to main)")
                else:
                    # POKA-YOKE BLOCK: Content differs and commits not in origin/main
                    raise RuntimeError(
                        f"[POKA-YOKE] Branch '{branch_name}' is NOT merged into origin/main. "
                        f"Feature commit {feature_commit[:8]} not found in remote main. "
                        "PR and Merge must complete first on GitHub. "
                        "If you squash merged, use force=True to bypass this check."
                    )


        # ALWAYS checkout main first
        self.checkout("main")
        
        # Pull latest main
        self.pull("origin", "main")
        
        # Delete local branch
        try:
            self.delete_local_branch(branch_name, force=True)
        except RuntimeError:
            # Maybe already deleted? Or we are on it? No, we switched to main. 
            # If force=True fails, something weird is up. Raising is fine.
            raise

        # Delete remote branch
        try:
            self.delete_remote_branch(branch_name)
        except Exception:
            # Remote branch might already be deleted, that's okay
            pass
        
        return f"Finished feature {branch_name}. Verified merge, deleted local/remote branches, and synced main."
        

