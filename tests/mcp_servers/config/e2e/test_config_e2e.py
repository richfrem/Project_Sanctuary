"""
Config MCP E2E Tests - Protocol Verification
============================================

Verifies all tools via JSON-RPC protocol against the real Config server.

CALLING EXAMPLES:
-----------------
pytest tests/mcp_servers/config/e2e/test_config_e2e.py -v -s

MCP TOOLS TESTED:
-----------------
| Tool              | Type  | Description              |
|-------------------|-------|--------------------------|
| config_list       | READ  | List config files        |
| config_read       | READ  | Read config content      |
| config_write      | WRITE | Create/Update config     |
| config_delete     | WRITE | Delete config file       |

"""
import pytest
import os
import json
from pathlib import Path
from tests.mcp_servers.base.base_e2e_test import BaseE2ETest

PROJECT_ROOT = Path(__file__).resolve().parents[4]

@pytest.mark.e2e
class TestConfigE2E(BaseE2ETest):
    """
    E2E Tests for Config MCP Server.
    Verifies tool execution via MCP protocol.
    """
    SERVER_NAME = "config"
    SERVER_MODULE = "mcp_servers.config.server"

    def test_config_lifecycle(self, mcp_client):
        """Test full cycle: List -> Write -> Read -> Delete"""
        
        print("\n=== Config E2E Lifecycle Verification ===")

        # 1. Verify Tools
        tools = mcp_client.list_tools()
        names = [t["name"] for t in tools]
        print(f"✅ Tools Available: {names}")
        assert "config_write" in names
        assert "config_read" in names
        assert "config_list" in names
        assert "config_delete" in names

        # 2. Write Config
        filename = "e2e_test_config.json"
        content = {"e2e_status": "running", "verified": True}
        content_str = json.dumps(content)
        
        write_res = mcp_client.call_tool("config_write", {
            "filename": filename,
            "content": content_str
        })
        write_text = write_res.get("content", [])[0]["text"]
        print(f"\n🆕 config_write: {write_text}")
        assert "success" in write_text.lower() or "written" in write_text.lower()

        try:
            # 3. Read Config
            read_res = mcp_client.call_tool("config_read", {"filename": filename})
            read_text = read_res.get("content", [])[0]["text"]
            print(f"📄 config_read: Retrieved content length {len(read_text)}")
            
            read_data = json.loads(read_text)
            assert read_data == content
            print("   ✅ Content verified")

            # 4. List Configs
            list_res = mcp_client.call_tool("config_list", {})
            list_text = list_res.get("content", [])[0]["text"]
            print(f"📋 config_list: Output received") 
            assert filename in list_text or "e2e_test_config.json" in list_text
            print("   ✅ Created file found in list")

        finally:
            # 5. Delete Config (Cleanup via Tool)
            # Config server HAS specific delete tool!
            delete_res = mcp_client.call_tool("config_delete", {"filename": filename})
            delete_text = delete_res.get("content", [])[0]["text"]
            print(f"🗑️ config_delete: {delete_text}")
            assert "deleted" in delete_text.lower()
