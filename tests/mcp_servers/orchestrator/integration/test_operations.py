"""
Orchestrator MCP Integration Tests - Operations Testing
=====================================================

Integration tests for Orchestrator MCP.
Verifies the tools function correctly and call expected dependencies.

MCP OPERATIONS:
---------------
| Operation                        | Type | Description                          |
|----------------------------------|------|--------------------------------------|
| orchestrator_dispatch_mission    | WRITE| Dispatch task (Mock)                 |
| orchestrator_run_strategic_cycle | ACTION| Run full cycle (mocks RAG/Cortex)   |

"""
import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

# Import the functions to test directly from server.py (since they are decorated tools)
from mcp_servers.orchestrator.server import (
    orchestrator_dispatch_mission,
    orchestrator_run_strategic_cycle
)


def test_dispatch_mission():
    """Test dispatch_mission tool (Basic)."""
    res = orchestrator_dispatch_mission(
        mission_id="M-001",
        objective="Test Objective",
        assigned_agent="TestAgent"
    )
    assert "M-001" in res
    assert "TestAgent" in res
    assert "Test Objective" in res


@patch("mcp_servers.orchestrator.server.CortexOperations")
def test_strategic_cycle(mock_cortex_cls, tmp_path):
    """
    Test orchestrator_run_strategic_cycle.
    Mocks CortexOperations to avoid real DB interaction.
    """
    # Setup mock
    mock_ops = MagicMock()
    mock_cortex_cls.return_value = mock_ops
    
    msg = "Unit Test"
    mock_ops.ingest_incremental.return_value = {"status": "success", "added": 1}
    mock_ops.guardian_wakeup.return_value = {"status": "success", "digest": "path/to/digest"}
    
    # Run tool
    report_path = str(tmp_path / "report.md")
    with open(report_path, "w") as f:
        f.write("# Report")
        
    result = orchestrator_run_strategic_cycle(
        gap_description="Testing Gap",
        research_report_path=report_path,
        days_to_synthesize=1
    )
    
    print("\n📦 Cycle Result:\n", result)
    
    # Assertions
    assert "Strategic Crucible Cycle" in result
    assert "Ingesting Report" in result
    assert "Generating Adaptation Packet" in result
    assert "Waking Guardian Cache" in result
    assert "Cycle Complete" in result
    
    # Verify calls
    mock_ops.ingest_incremental.assert_called_once()
    mock_ops.guardian_wakeup.assert_called_once()
