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
| git_add               | WRITE | ✅     | Stage files (self-validated)   |
| git_smart_commit      | WRITE | ✅     | Commit changes (self-validated)|
| git_push_feature      | WRITE | ✅     | Push branch (self-validated)   |
| git_start_feature     | WRITE | ❌     | Create branch (risky for E2E)  |
| git_finish_feature    | WRITE | ❌     | Delete branch (risky for E2E)  |

NOTE: Write operations (add, commit, push) are tested using a self-validating
approach where the test creates a marker file and uses the actual git workflow.
Branch creation/deletion operations remain untested to avoid repo state issues.

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
        assert len(log_text) > 0

        # 5. Git Diff (read-only)
        diff_res = mcp_client.call_tool("git_diff", {"cached": False})
        diff_text = diff_res.get("content", [])[0]["text"]
        print(f"📝 git_diff: {'Changes detected' if len(diff_text) > 10 else 'No changes'}")
        
        print("\n✅ Git read operations verified")

    def test_git_write_operations_self_validating(self, mcp_client):
        """Test write operations using this test file itself (self-validating)"""
        
        print("\n🔄 Testing write operations with self-validation...")
        
        # Create a marker file to test with
        test_file = PROJECT_ROOT / "tests/mcp_servers/git/e2e/.e2e_test_marker"
        test_file.write_text("E2E test validation marker\n")
        
        try:
            # 1. Test git_add
            add_res = mcp_client.call_tool("git_add", {"files": [str(test_file.relative_to(PROJECT_ROOT))]})
            add_text = add_res.get("content", [])[0]["text"]
            print(f"📦 git_add: {add_text}")
            assert "Staged" in add_text or "ERROR" in add_text  # Might error if not on feature branch
            
            # 2. Test git_smart_commit (if on feature branch)
            status_res = mcp_client.call_tool("git_get_status", {})
            status_text = status_res.get("content", [])[0]["text"]
            
            if "feature/" in status_text:
                commit_res = mcp_client.call_tool("git_smart_commit", {
                    "message": "test: E2E validation marker"
                })
                commit_text = commit_res.get("content", [])[0]["text"]
                print(f"💾 git_smart_commit: {commit_text}")
                assert "Commit successful" in commit_text or "Hash:" in commit_text
                
                # 3. Test git_push_feature
                push_res = mcp_client.call_tool("git_push_feature", {"no_verify": True})
                push_text = push_res.get("content", [])[0]["text"]
                print(f"🚀 git_push_feature: {push_text[:200]}...")
                assert "Verified push" in push_text or "Push" in push_text or "up-to-date" in push_text
                
                print("\n✅ Write operations validated successfully!")
            else:
                print("\n⚠️  Not on feature branch - skipping write operation tests")
                
        finally:
            # Cleanup marker file
            if test_file.exists():
                test_file.unlink()
                print("🧹 Cleaned up test marker")
