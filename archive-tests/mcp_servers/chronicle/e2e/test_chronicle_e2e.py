"""
Chronicle MCP E2E Tests - Protocol Verification
===============================================

Verifies all tools via JSON-RPC protocol against the real Chronicle server.

CALLING EXAMPLES:
-----------------
pytest tests/mcp_servers/chronicle/e2e/test_chronicle_e2e.py -v -s

MCP TOOLS TESTED:
-----------------
| Tool                  | Type  | Description              |
|-----------------------|-------|--------------------------|
| chronicle_create_entry| WRITE | Create entry             |
| chronicle_get_entry   | READ  | Get entry                |
| chronicle_list_entries| READ  | List entries             |
| chronicle_search      | READ  | Search entries           |
| chronicle_update_entry| WRITE | Update entry             |

"""
import pytest
import os
import re
from pathlib import Path
from tests.mcp_servers.base.base_e2e_test import BaseE2ETest

PROJECT_ROOT = Path(__file__).resolve().parents[4]

@pytest.mark.e2e
class TestChronicleE2E(BaseE2ETest):
    SERVER_NAME = "chronicle"
    SERVER_MODULE = "mcp_servers.chronicle.server"

    def test_chronicle_lifecycle(self, mcp_client):
        """Test full cycle: Create -> Get -> List -> Search -> Update -> Cleanup"""
        
        # 1. Verify Tools
        tools = mcp_client.list_tools()
        names = [t["name"] for t in tools]
        print(f"✅ Tools Available: {names}")
        assert "chronicle_create_entry" in names

        # 2. Create Entry
        title = "E2E Test Entry"
        create_res = mcp_client.call_tool("chronicle_create_entry", {
            "request": {
                "title": title,
                "content": "Test Content",
                "author": "E2E Bot",
                "classification": "public",
                "status": "draft"
            }
        })
        create_text = create_res.get("content", [])[0]["text"]
        print(f"\n🆕 Create: {create_text}")
        
        # Parse return: "Created Chronicle Entry {number}: {path}"
        match = re.search(r"Created Chronicle Entry (\d+): (.+)", create_text)
        assert match, f"Failed to parse creation response: {create_text}"
        
        entry_number = int(match.group(1))
        returned_path = match.group(2).strip()
        
        if Path(returned_path).is_absolute():
            entry_path = Path(returned_path)
        else:
            entry_path = PROJECT_ROOT / returned_path
            
        print(f"   Entry #{entry_number} at {entry_path}")

        try:
            # 3. Get Entry
            get_res = mcp_client.call_tool("chronicle_get_entry", {"request": {"entry_number": entry_number}})
            get_text = get_res.get("content", [])[0]["text"]
            assert f"Entry {entry_number}" in get_text
            assert title in get_text
            print("📄 chronicle_get_entry: Verified content")

            # 4. List Entries
            list_res = mcp_client.call_tool("chronicle_list_entries", {"request": {"limit": 5}})
            list_text = list_res.get("content", [])[0]["text"]
            # Format: "- {03d}: {title}..."
            assert f"{entry_number:03d}" in list_text or f"{entry_number}:" in list_text
            print(f"📋 chronicle_list_entries: Verified listing")

            # 5. Search
            search_res = mcp_client.call_tool("chronicle_search", {"request": {"query": "Test Entry"}})
            search_text = search_res.get("content", [])[0]["text"]
            assert f"{entry_number:03d}" in search_text or f"{entry_number}:" in search_text
            print(f"🔍 chronicle_search: Found entry")

            # 6. Update
            update_res = mcp_client.call_tool("chronicle_update_entry", {
                "request": {
                    "entry_number": entry_number,
                    "updates": {"status": "deprecated"},
                    "reason": "Test Cleanup"
                }
            })
            update_text = update_res.get("content", [])[0]["text"]
            print(f"🔄 chronicle_update_entry: {update_text}")
            assert "deprecated" in update_text or "Fields: status" in update_text

        finally:
            # 7. Cleanup
            if entry_path.exists():
                os.remove(entry_path)
                print(f"🧹 Cleaned up {entry_path}")
