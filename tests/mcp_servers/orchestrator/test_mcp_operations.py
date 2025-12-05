"""
Unit tests for Orchestrator MCP server operations.

Tests the actual MCP tools exposed by server.py:
- orchestrator_dispatch_mission
- orchestrator_run_strategic_cycle
"""
import pytest
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the MCP server module
from mcp_servers.orchestrator import server


class TestOrchestratorMCPOperations:
    """Test the MCP operations exposed by the Orchestrator server."""
    
    def test_orchestrator_dispatch_mission_basic(self):
        """Test orchestrator_dispatch_mission with basic parameters."""
        result = server.orchestrator_dispatch_mission(
            mission_id="TEST_001",
            objective="Test mission objective",
            assigned_agent="Kilo"
        )
        
        assert isinstance(result, str)
        assert "TEST_001" in result
        assert "Kilo" in result
        assert "Test mission objective" in result
        assert "dispatched" in result.lower()
    
    def test_orchestrator_dispatch_mission_custom_agent(self):
        """Test orchestrator_dispatch_mission with custom agent."""
        result = server.orchestrator_dispatch_mission(
            mission_id="TEST_002",
            objective="Custom agent test",
            assigned_agent="CustomAgent"
        )
        
        assert "CustomAgent" in result
        assert "TEST_002" in result
    
    def test_orchestrator_dispatch_mission_default_agent(self):
        """Test orchestrator_dispatch_mission uses default agent."""
        result = server.orchestrator_dispatch_mission(
            mission_id="TEST_003",
            objective="Default agent test"
        )
        
        # Default agent is "Kilo"
        assert "Kilo" in result
    
    @patch('mcp_servers.orchestrator.server.CortexOperations')
    def test_orchestrator_run_strategic_cycle_success(self, mock_cortex_ops):
        """Test orchestrator_run_strategic_cycle with successful execution."""
        # Mock the CortexOperations instance
        mock_instance = MagicMock()
        mock_instance.ingest_incremental.return_value = {"status": "success", "documents_added": 1}
        mock_instance.guardian_wakeup.return_value = {"status": "success", "cache_updated": True}
        mock_cortex_ops.return_value = mock_instance
        
        result = server.orchestrator_run_strategic_cycle(
            gap_description="Test strategic gap",
            research_report_path="test_report.md",
            days_to_synthesize=1
        )
        
        assert isinstance(result, str)
        assert "Strategic Crucible Cycle" in result
        assert "Test strategic gap" in result
        assert "Ingesting Report" in result
        assert "test_report.md" in result
        assert "Cycle Complete" in result
        
        # Verify CortexOperations was called correctly
        mock_instance.ingest_incremental.assert_called_once_with(["test_report.md"])
        mock_instance.guardian_wakeup.assert_called_once()
    
    @patch('mcp_servers.orchestrator.server.CortexOperations')
    def test_orchestrator_run_strategic_cycle_ingestion_failure(self, mock_cortex_ops):
        """Test orchestrator_run_strategic_cycle handles ingestion failure."""
        # Mock ingestion failure
        mock_instance = MagicMock()
        mock_instance.ingest_incremental.side_effect = Exception("Ingestion failed")
        mock_cortex_ops.return_value = mock_instance
        
        result = server.orchestrator_run_strategic_cycle(
            gap_description="Test gap",
            research_report_path="test.md"
        )
        
        assert "CRITICAL FAIL" in result
        assert "Ingestion failed" in result
    
    @patch('mcp_servers.orchestrator.server.CortexOperations')
    def test_orchestrator_run_strategic_cycle_cache_failure_non_critical(self, mock_cortex_ops):
        """Test orchestrator_run_strategic_cycle continues on cache failure."""
        # Mock successful ingestion but failed cache update
        mock_instance = MagicMock()
        mock_instance.ingest_incremental.return_value = {"status": "success"}
        mock_instance.guardian_wakeup.side_effect = Exception("Cache error")
        mock_cortex_ops.return_value = mock_instance
        
        result = server.orchestrator_run_strategic_cycle(
            gap_description="Test gap",
            research_report_path="test.md"
        )
        
        # Should complete despite cache failure
        assert "Cycle Complete" in result
        assert "WARN" in result
        assert "Cache error" in result
    
    @patch('mcp_servers.orchestrator.server.CortexOperations')
    def test_orchestrator_run_strategic_cycle_custom_days(self, mock_cortex_ops):
        """Test orchestrator_run_strategic_cycle with custom synthesis window."""
        mock_instance = MagicMock()
        mock_instance.ingest_incremental.return_value = {"status": "success"}
        mock_instance.guardian_wakeup.return_value = {"status": "success"}
        mock_cortex_ops.return_value = mock_instance
        
        result = server.orchestrator_run_strategic_cycle(
            gap_description="Test gap",
            research_report_path="test.md",
            days_to_synthesize=7
        )
        
        assert "7 days" in result
        assert "Cycle Complete" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
