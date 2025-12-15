"""
Protocol MCP E2E Tests - Protocol Verification
==============================================

Verifies all tools via JSON-RPC protocol against the real Protocol server.

CALLING EXAMPLES:
-----------------
pytest tests/mcp_servers/protocol/e2e/test_protocol_e2e.py -v -s

MCP TOOLS TESTED:
-----------------
| Tool              | Type  | Description              |
|-------------------|-------|--------------------------|
| protocol_list     | READ  | List protocols           |
| protocol_create   | WRITE | Create protocol          |
| protocol_get      | READ  | Get protocol content     |
| protocol_update   | WRITE | Update protocol          |
| protocol_search   | READ  | Search protocols         |

"""
import pytest
import os
import re
from pathlib import Path
from tests.mcp_servers.base.base_e2e_test import BaseE2ETest

PROJECT_ROOT = Path(__file__).resolve().parents[4]

@pytest.mark.e2e
class TestProtocolE2E(BaseE2ETest):
    """
    E2E Tests for Protocol MCP Server.
    Verifies tool execution via MCP protocol.
    """
    SERVER_NAME = "protocol"
    SERVER_MODULE = "mcp_servers.protocol.server"

    def test_protocol_lifecycle(self, mcp_client):
        """Test full cycle: List -> Create -> Get -> Update -> Search -> Cleanup"""
        
        # 1. Verify Tools
        tools = mcp_client.list_tools()
        names = [t["name"] for t in tools]
        print(f"✅ Tools Available: {names}")
        for t in ["protocol_list", "protocol_create", "protocol_get", "protocol_update", "protocol_search"]:
            assert t in names

        # 2. Create Protocol
        # Use high number to avoid conflict.
        number = 999
        title = "E2E Test Protocol"
        
        create_res = mcp_client.call_tool("protocol_create", {
            "number": number,
            "title": title,
            "status": "PROPOSED",
            "classification": "Test",
            "version": "0.1",
            "authority": "E2E",
            "content": "Test content."
        })
        # Protocol create returns string "Created Protocol 999: path"
        create_text = create_res.get("content", [])[0]["text"]
        print(f"\n🆕 protocol_create: {create_text}")
        
        # Parse path
        match = re.search(r"Created Protocol (\d+): (.+)", create_text)
        assert match, f"Failed to parse creation response: {create_text}"
        p_num = int(match.group(1))
        # The path in output might be relative or absolute.
        # Usually it returns '01_PROTOCOLS/999_E2E_Test_Protocol.md'
        path_str = match.group(2).strip()
        p_path = Path(PROJECT_ROOT) / path_str
        assert p_num == number

        try:
            # 3. Get Protocol
            get_res = mcp_client.call_tool("protocol_get", {"number": number})
            get_text = get_res.get("content", [])[0]["text"]
            assert title in get_text
            assert "Status: PROPOSED" in get_text
            print("📄 protocol_get: Verified content")

            # 4. Search
            search_res = mcp_client.call_tool("protocol_search", {"query": "Test Protocol"})
            search_text = search_res.get("content", [])[0]["text"]
            assert f"{number}: {title}" in search_text
            print(f"🔍 protocol_search: Found protocol")

            # 5. Update
            update_res = mcp_client.call_tool("protocol_update", {
                "number": number,
                "updates": {"status": "DEPRECATED"},
                "reason": "Test Cleanup"
            })
            update_text = update_res.get("content", [])[0]["text"]
            print(f"🔄 protocol_update: {update_text}")
            assert "DEPRECATED" in update_text

        finally:
            # 6. Cleanup
            if p_path.exists():
                os.remove(p_path)
                print(f"🧹 Cleaned up {p_path}")
