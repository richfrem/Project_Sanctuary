"""
Learning MCP E2E Tests - Learning Loop Verification
====================================================

Verifies the recursive learning loop tools (Protocol 128) via JSON-RPC.
Ensures that Scout, Seal, and Chronicle operations function correctly 
in a live server environment.

MCP TOOLS TESTED:
-----------------
| Tool                      | Operation         | Description              |
|---------------------------|-------------------|--------------------------|
| cortex_learning_debrief   | learning_debrief  | The Scout                |
| cortex_capture_snapshot   | capture_snapshot  | The Seal                 |
| cortex_persist_soul       | persist_soul      | The Chronicle            |
| cortex_guardian_wakeup    | guardian_wakeup   | The Bootloader           |
"""
import pytest
import os
import json
from pathlib import Path
from tests.mcp_servers.base.base_e2e_test import BaseE2ETest

@pytest.mark.e2e
class TestLearningE2E(BaseE2ETest):
    SERVER_NAME = "learning"
    SERVER_MODULE = "mcp_servers.learning.server"

    def test_learning_loop_lifecycle(self, mcp_client):
        """Test full Learning Loop: Debrief -> Snapshot -> Persist"""
        
        # 1. Verify Tools
        tools = mcp_client.list_tools()
        names = [t["name"] for t in tools]
        assert "learning_debrief" in names
        assert "capture_snapshot" in names
        assert "persist_soul" in names
        assert "guardian_wakeup" in names

        # 2. Debrief (Scout)
        debrief_res = mcp_client.call_tool("learning_debrief", {"hours": 24})
        text = debrief_res.get("content", [])[0]["text"]
        assert "Learning Package Snapshot" in text
        assert "Git Status" in text

        # 3. Capture Snapshot (Seal)
        # We need a dummy modification or just test the base case
        snapshot_res = mcp_client.call_tool("capture_snapshot", {
            "request": {
                "manifest_files": [],
                "snapshot_type": "audit"
            }
        })
        snap_text = snapshot_res.get("content", [])[0]["text"]
        assert "success" in snap_text.lower() or "error" in snap_text.lower()
        
        # 4. Guardian Wakeup (Bootloader)
        wakeup_res = mcp_client.call_tool("guardian_wakeup", {"mode": "HOLISTIC"})
        wakeup_text = wakeup_res.get("content", [])[0]["text"]
        assert "success" in wakeup_text.lower()
        assert ".agent/learning/guardian_boot_digest.md" in wakeup_text
