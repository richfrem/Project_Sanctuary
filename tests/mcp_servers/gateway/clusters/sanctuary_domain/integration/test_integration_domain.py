"""
Integration tests for sanctuary_domain cluster.
Tests direct SSE communication (no Gateway).
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from integration_conftest import *


@pytest.mark.integration
class TestDomainClusterHealth:
    """Test sanctuary_domain cluster connectivity."""
    
    def test_cluster_health(self, domain_cluster):
        """Verify cluster health endpoint."""
        assert domain_cluster.health_check()
    
    def test_list_tools(self, domain_cluster):
        """Verify domain tools are exposed."""
        result = domain_cluster.list_tools()
        assert result["success"]
        
        tools = result["tools"]
        tool_names = [t["name"] for t in tools]
        
        expected_tools = ["chronicle-list-entries", "protocol-list", "task-list-tasks", "adr-list"]
        for tool in expected_tools:
            assert tool in tool_names, f"Missing tool: {tool}"


@pytest.mark.integration
class TestDomainChronicleTools:
    """Test sanctuary_domain chronicle tools via direct SSE."""
    
    def test_chronicle_list_entries(self, domain_cluster):
        """Test chronicle list via direct SSE."""
        result = domain_cluster.call_tool("chronicle-list-entries", {"limit": 5})
        assert result["success"], f"Tool call failed: {result.get('error')}"


@pytest.mark.integration
class TestDomainProtocolTools:
    """Test sanctuary_domain protocol tools via direct SSE."""
    
    def test_protocol_list(self, domain_cluster):
        """Test protocol list via direct SSE."""
        result = domain_cluster.call_tool("protocol-list", {})
        assert result["success"], f"Tool call failed: {result.get('error')}"


@pytest.mark.integration
class TestDomainTaskTools:
    """Test sanctuary_domain task tools via direct SSE."""
    
    def test_task_list_tasks(self, domain_cluster):
        """Test task list via direct SSE."""
        result = domain_cluster.call_tool("task-list-tasks", {})
        assert result["success"], f"Tool call failed: {result.get('error')}"


@pytest.mark.integration
class TestDomainADRTools:
    """Test sanctuary_domain ADR tools via direct SSE."""
    
    def test_adr_list(self, domain_cluster):
        """Test ADR list via direct SSE."""
        result = domain_cluster.call_tool("adr-list", {})
        assert result["success"], f"Tool call failed: {result.get('error')}"
