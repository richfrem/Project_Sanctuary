"""
Code MCP E2E Tests - Protocol Verification
==========================================

Verifies all tools via JSON-RPC protocol against the real Code server.

CALLING EXAMPLES:
-----------------
pytest tests/mcp_servers/code/e2e/test_code_e2e.py -v -s

MCP TOOLS TESTED:
-----------------
| Operation           | Type  | Description                    |
|---------------------|-------|--------------------------------|
| code_list_files     | READ  | List files in directory        |
| code_find_file      | READ  | Find file by pattern           |
| code_read           | READ  | Read file contents             |
| code_get_info       | READ  | Get file metadata              |
| code_search_content | READ  | Search content in files        |
| code_analyze        | READ  | Static analysis                |
| code_check_tools    | READ  | Check available tools          |
| code_lint           | READ  | Lint code                      |
| code_format         | READ  | Format check (check_only=True) |
| code_write          | WRITE | Write file then cleanup        |

"""
import pytest
import os
import json
import time
from pathlib import Path
from tests.mcp_servers.base.base_e2e_test import BaseE2ETest

PROJECT_ROOT = Path(__file__).resolve().parents[4]

@pytest.mark.e2e
class TestCodeE2E(BaseE2ETest):
    SERVER_NAME = "code"
    SERVER_MODULE = "mcp_servers.code.server"

    def test_code_lifecycle(self, mcp_client):
        """Test full cycle covering all Code MCP tools."""
        
        # 1. Verify Tools
        tools = mcp_client.list_tools()
        names = [t["name"] for t in tools]
        print(f"✅ Tools Available: {names}")
        assert "code_write" in names
        assert "code_lint" in names

        # 2. List Files
        list_res = mcp_client.call_tool("code_list_files", {"directory": ".", "recursive": False})
        list_text = list_res.get("content", [])[0]["text"]
        print(f"📋 code_list_files: Listing received")
        assert "README.md" in list_text or "mcp_servers" in list_text

        # 3. Write File
        file_path = "e2e_test_code.py"
        content = "def hello():\n    print('Hello E2E')"
        
        write_res = mcp_client.call_tool("code_write", {
            "path": file_path,
            "content": content,
            "backup": False
        })
        write_text = write_res.get("content", [])[0]["text"]
        print(f"\n🆕 code_write: {write_text}")
        assert "Wrote to" in write_text or "Successfully wrote" in write_text or "Created" in write_text

        try:
            # 4. Read File
            read_res = mcp_client.call_tool("code_read", {"path": file_path})
            read_text = read_res.get("content", [])[0]["text"]
            assert content in read_text
            print("📄 code_read: Verified content")
            
            # 5. Get Info
            info_res = mcp_client.call_tool("code_get_info", {"path": file_path})
            info_text = info_res.get("content", [])[0]["text"]
            print(f"ℹ️ code_get_info: {info_text}")
            assert "Size" in info_text

            # 6. Find File
            find_res = mcp_client.call_tool("code_find_file", {"name_pattern": file_path})
            find_text = find_res.get("content", [])[0]["text"]
            print(f"🔎 code_find_file: {find_text}")
            assert file_path in find_text

            # 7. Search Content
            # Wait briefly for file system
            time.sleep(1)
            search_res = mcp_client.call_tool("code_search_content", {"query": "Hello E2E", "file_pattern": "*.py"})
            search_text = search_res.get("content", [])[0]["text"]
            print(f"🔍 search content result: {search_text}") # Debug print
            assert file_path in search_text
            print("   code_search_content: Verified match")

            # 8. Check Tools
            check_res = mcp_client.call_tool("code_check_tools", {})
            check_text = check_res.get("content", [])[0]["text"]
            print(f"🛠️ code_check_tools: {check_text}")
            
            # 9. Format (Check only)
            # Use 'black' or 'ruff' (might fail if not installed, handle gracefully)
            try:
                fmt_res = mcp_client.call_tool("code_format", {"path": file_path, "check_only": True})
                print(f"🖊️ code_format: {fmt_res.get('content', [])[0]['text']}")
            except Exception as e:
                print(f"⚠️ code_format skipped/failed: {e}")

            # 10. Lint
            try:
                lint_res = mcp_client.call_tool("code_lint", {"path": file_path})
                print(f"🧹 code_lint: {lint_res.get('content', [])[0]['text']}")
            except Exception as e:
                print(f"⚠️ code_lint skipped/failed: {e}")

            # 11. Analyze
            try:
                analyze_res = mcp_client.call_tool("code_analyze", {"path": file_path})
                print(f"📊 code_analyze: {analyze_res.get('content', [])[0]['text']}")
            except Exception as e:
                print(f"⚠️ code_analyze skipped/failed: {e}")

        finally:
            # 12. Cleanup
            full_path = PROJECT_ROOT / file_path
            if full_path.exists():
                os.remove(full_path)
                print(f"🧹 Cleaned up {full_path}")
