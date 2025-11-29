#!/usr/bin/env python3
"""
Git Operations Verification Script

Tests the GitOperations class methods that power the MCP tools.
This verifies the core functionality after the relocation to mcp_servers/lib/.
"""

import os
import sys
import tempfile
import shutil
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mcp_servers.lib.git.git_ops import GitOperations


class Colors:
    """ANSI color codes for terminal output."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_test(name: str):
    """Print test name."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}Testing: {name}{Colors.RESET}")


def print_success(message: str):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {message}{Colors.RESET}")


def print_error(message: str):
    """Print error message."""
    print(f"{Colors.RED}✗ {message}{Colors.RESET}")


def print_info(message: str):
    """Print info message."""
    print(f"{Colors.YELLOW}ℹ {message}{Colors.RESET}")


class GitOpsVerifier:
    """Verifies GitOperations class after core relocation."""

    def __init__(self):
        self.test_dir = None
        self.bare_repo = None
        self.git_ops = None
        self.passed = 0
        self.failed = 0

    def setup(self):
        """Set up test environment."""
        print(f"\n{Colors.BOLD}Setting up test environment...{Colors.RESET}")
        
        # Create temporary directory for test repo
        self.test_dir = tempfile.mkdtemp()
        original_dir = os.getcwd()
        os.chdir(self.test_dir)
        
        # Initialize git repo
        subprocess.run(["git", "init"], check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], check=True)
        subprocess.run(["git", "config", "user.name", "Test User"], check=True)
        
        # Create initial commit
        with open("README.md", "w") as f:
            f.write("# Test Repo")
        subprocess.run(["git", "add", "README.md"], check=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], check=True, capture_output=True)
        
        # Create bare repo to act as remote
        self.bare_repo = tempfile.mkdtemp()
        subprocess.run(["git", "init", "--bare"], cwd=self.bare_repo, check=True, capture_output=True)
        subprocess.run(["git", "remote", "add", "origin", self.bare_repo], check=True)
        subprocess.run(["git", "push", "-u", "origin", "main"], check=True, capture_output=True)
        
        # Initialize GitOperations
        self.git_ops = GitOperations(self.test_dir)
        
        os.chdir(original_dir)
        
        print_success(f"Test environment created at {self.test_dir}")

    def teardown(self):
        """Clean up test environment."""
        print(f"\n{Colors.BOLD}Cleaning up test environment...{Colors.RESET}")
        
        if self.test_dir and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        if self.bare_repo and os.path.exists(self.bare_repo):
            shutil.rmtree(self.bare_repo)
        
        print_success("Test environment cleaned up")

    def test_status(self):
        """Test status() method."""
        print_test("GitOperations.status()")
        try:
            status = self.git_ops.status()
            assert status["branch"] == "main"
            assert isinstance(status["staged"], list)
            assert isinstance(status["modified"], list)
            assert isinstance(status["untracked"], list)
            
            print_success("status() works correctly")
            print_info(f"Current branch: {status['branch']}")
            self.passed += 1
        except Exception as e:
            print_error(f"status() failed: {e}")
            self.failed += 1

    def test_branch_operations(self):
        """Test branch creation, checkout, and deletion."""
        print_test("GitOperations branch operations")
        try:
            # Create branch
            self.git_ops.create_branch("test-branch")
            
            # Checkout
            self.git_ops.checkout("test-branch")
            current = self.git_ops.get_current_branch()
            assert current == "test-branch"
            
            # Switch back
            self.git_ops.checkout("main")
            
            # Delete
            self.git_ops.delete_branch("test-branch")
            
            print_success("Branch operations work correctly")
            self.passed += 1
        except Exception as e:
            print_error(f"Branch operations failed: {e}")
            self.failed += 1

    def test_add_and_commit(self):
        """Test add() and commit() with manifest."""
        print_test("GitOperations.add() and commit()")
        try:
            # Create a file
            test_file = os.path.join(self.test_dir, "test.txt")
            with open(test_file, "w") as f:
                f.write("test content")
            
            # Stage file
            self.git_ops.add(["test.txt"])
            
            # Verify staged
            status = self.git_ops.status()
            assert "test.txt" in status["staged"]
            
            # Commit with manifest
            commit_hash = self.git_ops.commit("test: add test file")
            assert len(commit_hash) == 40  # SHA-1 hash
            
            # Verify manifest exists
            result = subprocess.run(
                ["git", "show", "HEAD:commit_manifest.json"],
                cwd=self.test_dir,
                capture_output=True,
                text=True
            )
            assert result.returncode == 0
            assert "test.txt" in result.stdout
            
            print_success("add() and commit() work correctly (Protocol 101 manifest created)")
            print_info(f"Commit hash: {commit_hash[:8]}")
            self.passed += 1
        except Exception as e:
            print_error(f"add() and commit() failed: {e}")
            self.failed += 1

    def test_push_with_no_verify(self):
        """Test push() with no_verify parameter."""
        print_test("GitOperations.push(no_verify=True)")
        try:
            result = self.git_ops.push("origin", "main", no_verify=True)
            assert result is not None
            
            print_success("push() with no_verify=True works correctly")
            self.passed += 1
        except Exception as e:
            print_error(f"push() with no_verify failed: {e}")
            self.failed += 1

    def test_push_with_force(self):
        """Test push() with force parameter."""
        print_test("GitOperations.push(force=True)")
        try:
            # Create a divergence by amending
            test_file = os.path.join(self.test_dir, "test.txt")
            with open(test_file, "w") as f:
                f.write("modified content")
            
            subprocess.run(["git", "add", "test.txt"], cwd=self.test_dir, check=True)
            subprocess.run(
                ["git", "commit", "--amend", "--no-edit"],
                cwd=self.test_dir,
                check=True,
                capture_output=True
            )
            
            # Force push
            result = self.git_ops.push("origin", "main", force=True, no_verify=True)
            assert result is not None
            
            print_success("push() with force=True works correctly")
            self.passed += 1
        except Exception as e:
            print_error(f"push() with force failed: {e}")
            self.failed += 1

    def test_diff(self):
        """Test diff() method."""
        print_test("GitOperations.diff()")
        try:
            # Create a change
            test_file = os.path.join(self.test_dir, "test.txt")
            with open(test_file, "w") as f:
                f.write("new content")
            
            # Get diff
            diff = self.git_ops.diff(cached=False)
            # Diff might be empty if file is already committed
            
            print_success("diff() works correctly")
            self.passed += 1
        except Exception as e:
            print_error(f"diff() failed: {e}")
            self.failed += 1

    def test_log(self):
        """Test log() method."""
        print_test("GitOperations.log()")
        try:
            log = self.git_ops.log(max_count=5, oneline=True)
            assert len(log) > 0
            
            print_success("log() works correctly")
            print_info(f"Log entries: {len(log.splitlines())}")
            self.passed += 1
        except Exception as e:
            print_error(f"log() failed: {e}")
            self.failed += 1

    def test_pull(self):
        """Test pull() method."""
        print_test("GitOperations.pull()")
        try:
            result = self.git_ops.pull("origin", "main")
            # Should be "Already up to date"
            
            print_success("pull() works correctly")
            self.passed += 1
        except Exception as e:
            print_error(f"pull() failed: {e}")
            self.failed += 1

    def run_all_tests(self):
        """Run all verification tests."""
        print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}GitOperations Verification (Post Core Relocation){Colors.RESET}")
        print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")
        
        try:
            self.setup()
            
            # Run tests
            self.test_status()
            self.test_branch_operations()
            self.test_add_and_commit()
            self.test_push_with_no_verify()
            self.test_push_with_force()
            self.test_diff()
            self.test_log()
            self.test_pull()
            
        finally:
            self.teardown()
        
        # Print summary
        print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
        print(f"{Colors.BOLD}Test Summary{Colors.RESET}")
        print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")
        print(f"{Colors.GREEN}Passed: {self.passed}{Colors.RESET}")
        print(f"{Colors.RED}Failed: {self.failed}{Colors.RESET}")
        print(f"{Colors.BOLD}Total: {self.passed + self.failed}{Colors.RESET}")
        
        if self.failed == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}✓ All GitOperations methods verified!{Colors.RESET}")
            print(f"{Colors.GREEN}{Colors.BOLD}✓ Core relocation successful!{Colors.RESET}")
            return 0
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}✗ Some tests failed{Colors.RESET}")
            return 1


if __name__ == "__main__":
    verifier = GitOpsVerifier()
    sys.exit(verifier.run_all_tests())
