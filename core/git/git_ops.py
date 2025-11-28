import subprocess
import json
import hashlib
import os
import datetime
from typing import List, Dict, Any, Optional

class GitOperations:
    """
    Handles git operations with Protocol 101 enforcement.
    """
    
    def __init__(self, repo_path: str = ".", base_dir: Optional[str] = None):
        self.repo_path = os.path.abspath(repo_path)
        self.manifest_file = "commit_manifest.json"
        
        # Security: Restrict operations to base_dir if specified
        self.base_dir = os.path.abspath(base_dir) if base_dir else None
        if self.base_dir and not self.repo_path.startswith(self.base_dir):
            raise ValueError(f"Repository path {self.repo_path} is outside base directory {self.base_dir}")

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
            raise RuntimeError(f"Git command failed: {e.stderr}")

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

    def calculate_file_hash(self, filepath: str) -> str:
        """Calculate SHA256 hash of a file."""
        full_path = os.path.join(self.repo_path, filepath)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"File not found: {filepath}")
            
        sha256_hash = hashlib.sha256()
        with open(full_path, "rb") as f:
            # Read and update hash string value in blocks of 4K
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def generate_manifest(self) -> Dict[str, Any]:
        """
        Generate Protocol 101 manifest for staged files.
        Returns the manifest dictionary.
        """
        staged_files = self.get_staged_files()
        if not staged_files:
            raise ValueError("No files staged for commit.")

        manifest_entries = []
        for filepath in staged_files:
            # Skip the manifest itself if it's somehow staged already
            if filepath == self.manifest_file:
                continue
                
            file_hash = self.calculate_file_hash(filepath)
            manifest_entries.append({
                "path": filepath,
                "sha256": file_hash
            })

        manifest = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "author": "Guardian (Smart Git MCP)",
            "files": manifest_entries
        }
        return manifest

    def commit(self, message: str) -> str:
        """
        Commit staged files with P101 manifest.
        1. Generate manifest.
        2. Write manifest to disk.
        3. Stage manifest.
        4. Commit.
        Returns commit hash.
        """
        # 1. Generate Manifest
        manifest = self.generate_manifest()
        
        # 2. Write Manifest
        manifest_path = os.path.join(self.repo_path, self.manifest_file)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
            
        # 3. Stage Manifest
        self._run_git(["add", self.manifest_file])
        
        # 4. Commit
        # We assume the pre-commit hook is active. 
        # Since we are generating a valid manifest, we do NOT need IS_MCP_AGENT=1 bypass.
        # However, if we are running in an environment where the hook might block us 
        # for other reasons, we should be aware. 
        # For now, we commit normally.
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

    def push(self, remote: str = "origin", branch: str = None) -> str:
        """Push to remote."""
        if branch is None:
            branch = self.get_current_branch()
        return self._run_git(["push", remote, branch])

    def pull(self, remote: str = "origin", branch: str = None) -> str:
        """Pull from remote."""
        if branch is None:
            branch = self.get_current_branch()
        return self._run_git(["pull", remote, branch])

    def delete_branch(self, branch_name: str, force: bool = False) -> None:
        """Delete a branch."""
        flag = "-D" if force else "-d"
        self._run_git(["branch", flag, branch_name])

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

    def create_pr(self, title: str, body: str = "", base: str = "main") -> str:
        """
        Create a GitHub Pull Request using gh CLI.
        Requires: gh CLI installed and authenticated.
        """
        try:
            current_branch = self.get_current_branch()
            result = subprocess.run(
                ["gh", "pr", "create", "--title", title, "--body", body, "--base", base, "--head", current_branch],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"GitHub CLI command failed: {e.stderr}")
        except FileNotFoundError:
            raise RuntimeError("GitHub CLI (gh) not found. Install it with: brew install gh")


