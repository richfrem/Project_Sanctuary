"""
Integration tests for sanctuary_domain cluster.
Tests direct SSE communication via MCP SDK.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from integration_conftest import *


@pytest.mark.integration
@pytest.mark.asyncio
class TestDomainClusterHealth:
    """Test sanctuary_domain cluster connectivity."""
    
    async def test_cluster_health(self, domain_cluster):
        """Verify cluster health endpoint."""
        assert await domain_cluster.health_check()
    
    async def test_list_tools(self, domain_cluster):
        """Verify domain tools are exposed."""
        result = await domain_cluster.list_tools()
        tool_names = [t.name for t in result.tools]
        
        expected_tools = ["chronicle-list-entries", "protocol-list", "task-list-tasks", "adr-list"]
        for tool in expected_tools:
            assert tool in tool_names, f"Missing tool: {tool}"


@pytest.mark.integration
@pytest.mark.asyncio
class TestDomainChronicleTools:
    """Test sanctuary_domain chronicle tools via direct SSE."""
    
    async def test_chronicle_list_entries(self, domain_cluster):
        """Test chronicle list via direct SSE."""
        result = await domain_cluster.call_tool("chronicle-list-entries", {"limit": 5})
        assert len(result.content) > 0
        assert "Chronicle Entries" in result.content[0].text or "entries" in result.content[0].text


@pytest.mark.integration
@pytest.mark.asyncio
class TestDomainProtocolTools:
    """Test sanctuary_domain protocol tools via direct SSE."""
    
    async def test_protocol_list(self, domain_cluster):
        """Test protocol list via direct SSE."""
        result = await domain_cluster.call_tool("protocol-list", {})
        assert len(result.content) > 0
        assert "Protocol" in result.content[0].text


@pytest.mark.integration
@pytest.mark.asyncio
class TestDomainTaskTools:
    """Test sanctuary_domain task tools via direct SSE."""
    
    async def test_task_list_tasks(self, domain_cluster):
        """Test task list via direct SSE."""
        result = await domain_cluster.call_tool("task-list-tasks", {})
        assert len(result.content) > 0
        assert "tasks" in result.content[0].text


@pytest.mark.integration
@pytest.mark.asyncio
class TestDomainADRTools:
    """Test sanctuary_domain ADR tools via direct SSE."""
    
    async def test_adr_list(self, domain_cluster):
        """Test ADR list via direct SSE."""
        result = await domain_cluster.call_tool("adr-list", {})
        assert len(result.content) > 0
        # ADR directory might be empty, so checking content presence is enough
