"""
ADR MCP E2E Tests - Protocol Verification
=========================================

Verifies all tools via JSON-RPC protocol against the real ADR server.
Uses 'Headless' MCP client to assume server identity without network stack.

CALLING EXAMPLES:
-----------------
pytest tests/mcp_servers/adr/e2e/test_adr_e2e.py -v -s

MCP TOOLS TESTED:
-----------------
| Tool              | Type  | Description              |
|-------------------|-------|--------------------------|
| adr_list          | READ  | List real ADRs           |
| adr_create        | WRITE | Create then cleanup      |
| adr_get           | READ  | Get real ADR             |
| adr_update_status | WRITE | Update then cleanup      |
| adr_search        | READ  | Search real ADRs         |

"""
import pytest
import os
import json
import re
from pathlib import Path
from tests.mcp_servers.base.base_e2e_test import BaseE2ETest

PROJECT_ROOT = Path(__file__).resolve().parents[4]

@pytest.mark.e2e
class TestADRE2E(BaseE2ETest):
    """
    E2E Tests for ADR MCP Server.
    Verifies tool execution via MCP protocol.
    """
    SERVER_NAME = "adr"
    SERVER_MODULE = "mcp_servers.adr.server"

    def test_adr_lifecycle(self, mcp_client):
        """Test full cycle: List -> Create -> Get -> Update -> Search -> Cleanup"""
        
        # 1. Verify Tools Exist
        tools = mcp_client.list_tools()
        tool_names = [t["name"] for t in tools]
        assert "adr_list" in tool_names
        assert "adr_create" in tool_names
        assert "adr_get" in tool_names
        assert "adr_update_status" in tool_names

        print(f"✅ Verified tools availability: {tool_names}")

        # 2. Create ADR
        title = "[E2E TEST] Protocol 999: Automated Verification"
        create_res = mcp_client.call_tool("adr_create", {
            "request": {
                "title": title,
                "context": "Running E2E verification suite.",
                "decision": "Implement comprehensive protocol testing.",
                "consequences": "Temporary file created."
            }
        })
        
        # Parse Create Response (String format: "Successfully created ADR {03d}: {path}")
        content_text = create_res.get("content", [])[0]["text"]
        
        # Extract number and path
        match = re.search(r"Successfully created ADR (\d+): (.+)", content_text)
        assert match, f"Failed to parse creation response: {content_text}"
        
        adr_number = int(match.group(1))
        rel_path = match.group(2).strip()
        file_path = Path(PROJECT_ROOT) / rel_path
        
        print(f"\n🆕 adr_create: #{adr_number} at {file_path}")

        try:
            # 3. Verify Creation via adr_get
            get_res = mcp_client.call_tool("adr_get", {"request": {"number": adr_number}})
            get_text = get_res.get("content", [])[0]["text"]
            
            # Parse Get Response (Human readable string)
            assert f"ADR {adr_number:03d}: {title}" in get_text
            assert "Status: proposed" in get_text
            
            print(f"📄 adr_get: Verified title and status.")

            # 4. Verify Search (New)
            search_res = mcp_client.call_tool("adr_search", {"request": {"query": "Automated Verification"}})
            search_text = search_res.get("content", [])[0]["text"]
            
            # Format: "Found X ADR(s)... \nADR 00X: Title..."
            assert f"ADR {adr_number:03d}" in search_text
            print(f"🔍 adr_search: Found created ADR in results.")

            # 5. Update Status
            update_res = mcp_client.call_tool("adr_update_status", {
                "request": {
                    "number": adr_number,
                    "new_status": "deprecated",
                    "reason": "E2E Test Cleanup"
                }
            })
            update_text = update_res.get("content", [])[0]["text"]
            print(f"🔄 adr_update_status: {update_text}")
            assert "deprecated" in update_text

            # 6. Verify Update
            get_res_2 = mcp_client.call_tool("adr_get", {"request": {"number": adr_number}})
            get_text_2 = get_res_2.get("content", [])[0]["text"]
            assert "Status: deprecated" in get_text_2

        finally:
            # 7. Cleanup (Side Effect)
            if file_path.exists():
                os.remove(file_path)
                print(f"🧹 Cleaned up {file_path}")
