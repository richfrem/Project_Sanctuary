"""
Git Workflow MCP E2E Tests - Protocol Verification
==================================================

Verifies all tools via JSON-RPC protocol against the real Git Workflow server.
Uses read-only operations to avoid affecting repository state.

CALLING EXAMPLES:
-----------------
pytest tests/mcp_servers/git/e2e/test_git_e2e.py -v -s

MCP TOOLS TESTED:
-----------------
| Tool                  | Type  | Tested | Description                    |
|-----------------------|-------|--------|--------------------------------|
| git_get_safety_rules  | READ  | ✅     | Get safety protocols           |
| git_get_status        | READ  | ✅     | Get repository status          |
| git_log               | READ  | ✅     | View commit history            |
| git_diff              | READ  | ✅     | View changes                   |
| git_add               | WRITE | ❌     | Stage files (risky for E2E)    |
| git_smart_commit      | WRITE | ❌     | Commit changes (risky for E2E) |
| git_push_feature      | WRITE | ❌     | Push branch (risky for E2E)    |
| git_start_feature     | WRITE | ❌     | Create branch (risky for E2E)  |
| git_finish_feature    | WRITE | ❌     | Delete branch (risky for E2E)  |

NOTE: Write operations are NOT tested in E2E to avoid affecting real repository
state. These are covered in integration tests with isolated test repositories.

"""
import pytest
from pathlib import Path
from tests.mcp_servers.base.base_e2e_test import BaseE2ETest

PROJECT_ROOT = Path(__file__).resolve().parents[4]

@pytest.mark.e2e
class TestGitWorkflowE2E(BaseE2ETest):
    SERVER_NAME = "git"
    SERVER_MODULE = "mcp_servers.git.server"

    def test_git_read_operations(self, mcp_client):
        """Test read-only Git operations (safe for E2E)"""
        
        # 1. Verify Tools
        tools = mcp_client.list_tools()
        names = [t["name"] for t in tools]
        print(f"✅ Tools Available: {names}")
        
        assert "git_get_status" in names
        assert "git_get_safety_rules" in names

        # 2. Get Safety Rules
        rules_res = mcp_client.call_tool("git_get_safety_rules", {})
        rules_text = rules_res.get("content", [])[0]["text"]
        print(f"\n📜 git_get_safety_rules: {rules_text[:200]}...")
        assert "MAIN IS PROTECTED" in rules_text or "safety" in rules_text.lower()

        # 3. Get Status
        status_res = mcp_client.call_tool("git_get_status", {})
        status_text = status_res.get("content", [])[0]["text"]
        print(f"📊 git_get_status: {status_text}")
        assert "Branch:" in status_text or "branch" in status_text.lower()

        # 4. Git Log (read-only)
        log_res = mcp_client.call_tool("git_log", {"max_count": 3, "oneline": True})
        log_text = log_res.get("content", [])[0]["text"]
        print(f"📜 git_log: {log_text[:200]}...")
        # Should have some commit history
        assert len(log_text) > 0

        # 5. Git Diff (read-only, might be empty)
        diff_res = mcp_client.call_tool("git_diff", {"cached": False})
        diff_text = diff_res.get("content", [])[0]["text"]
        print(f"📝 git_diff: {'Changes detected' if len(diff_text) > 10 else 'No changes'}")
        
        print("\n✅ Git read operations verified")
        print("⚠️  Write operations (commit, push, branch) not tested in E2E")
        print("   (Covered in integration tests to avoid repo state changes)")
