"""
Orchestrator MCP E2E Tests - Protocol Verification
==================================================

Verifies all tools via JSON-RPC protocol against the real Orchestrator server.
Minimal test - full orchestration patterns preserved in:
- test_protocol_056_headless.py (Triple Loop validation)
- test_strategic_crucible.py (Strategic Crucible Loop)

CALLING EXAMPLES:
-----------------
pytest tests/mcp_servers/orchestrator/e2e/test_orchestrator_e2e.py -v -s

MCP TOOLS TESTED:
-----------------
| Tool                           | Type  | Description                    |
|--------------------------------|-------|--------------------------------|
| orchestrator_dispatch_mission  | WRITE | Dispatch mission to agent      |
| orchestrator_run_strategic_cycle | WRITE | Run strategic crucible loop  |

NOTE: This test verifies tool availability only. Full orchestration patterns
are tested in test_protocol_056_headless.py and test_strategic_crucible.py.
These are preserved but not run by default due to slow LLM operations.

"""
import pytest
from pathlib import Path
from tests.mcp_servers.base.base_e2e_test import BaseE2ETest

PROJECT_ROOT = Path(__file__).resolve().parents[4]

@pytest.mark.e2e
class TestOrchestratorE2E(BaseE2ETest):
    SERVER_NAME = "orchestrator"
    SERVER_MODULE = "mcp_servers.orchestrator.server"

    def test_orchestrator_tools_available(self, mcp_client):
        """Verify orchestrator tools are available (skip slow execution)"""
        
        # 1. Verify Tools
        tools = mcp_client.list_tools()
        names = [t["name"] for t in tools]
        print(f"✅ Tools Available: {names}")
        
        # Verify key tools exist
        assert "orchestrator_dispatch_mission" in names
        assert "orchestrator_run_strategic_cycle" in names
        
        print("\n✅ Orchestrator tools verified")
        print("⚠️  Full orchestration tests preserved in:")
        print("   - test_protocol_056_headless.py")
        print("   - test_strategic_crucible.py")
        print("   (Not run by default due to slow LLM operations)")
