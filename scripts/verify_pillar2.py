
import sys
import os
from unittest.mock import MagicMock

# Mock FastMCP before importing server
sys.modules["fastmcp"] = MagicMock()
mock_mcp = MagicMock()
def tool_decorator():
    def wrapper(func):
        return func
    return wrapper
mock_mcp.tool = tool_decorator
sys.modules["fastmcp"].FastMCP = MagicMock(return_value=mock_mcp)

# Add project root to path
sys.path.append(os.getcwd())

from mcp_servers.lib.git.git_ops import GitOperations

# Mock GitOperations to capture commands
original_run_git = GitOperations._run_git
command_log = []

def mock_run_git(self, args):
    cmd = "git " + " ".join(args)
    command_log.append(cmd)
    if args[0] == "rev-parse" and "--abbrev-ref" in args:
        return "feature/test-branch"
    return ""

GitOperations._run_git = mock_run_git

# Now import server which uses the mocked FastMCP and GitOperations
from mcp_servers.system.git_workflow.server import git_start_feature, git_smart_commit, git_finish_feature

print("=== TEST 1: Code Integrity (Start Feature) ===")
command_log.clear()
git_start_feature("999", "test-integrity")
for cmd in command_log:
    print(f"[EXEC] {cmd}")
    if "pull" in cmd or "reset" in cmd:
        print("!!! ALARM: UNAUTHORIZED SYNC DETECTED !!!")

print("\n=== TEST 2: Code Integrity (Smart Commit) ===")
command_log.clear()
git_smart_commit("test commit")
for cmd in command_log:
    print(f"[EXEC] {cmd}")
    if "pull" in cmd or "reset" in cmd:
        print("!!! ALARM: UNAUTHORIZED SYNC DETECTED !!!")

print("\n=== TEST 3: Finish Feature (Safety Check) ===")
command_log.clear()
git_finish_feature("feature/task-999-test-integrity")
for cmd in command_log:
    print(f"[EXEC] {cmd}")
