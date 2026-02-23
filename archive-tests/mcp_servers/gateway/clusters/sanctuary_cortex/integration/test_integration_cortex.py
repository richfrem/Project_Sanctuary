"""
Integration tests for sanctuary_cortex cluster.
Tests direct SSE communication via MCP SDK.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from integration_conftest import *


@pytest.mark.integration
@pytest.mark.asyncio
class TestCortexClusterHealth:
    """Test sanctuary_cortex cluster connectivity."""
    
    async def test_cluster_health(self, cortex_cluster):
        """Verify cluster health endpoint."""
        assert await cortex_cluster.health_check()
    
    async def test_list_tools(self, cortex_cluster):
        """Verify cortex tools are exposed."""
        result = await cortex_cluster.list_tools()
        tool_names = [t.name for t in result.tools]
        
        expected_tools = ["cortex-get-stats", "cortex-query"]
        for tool in expected_tools:
            assert tool in tool_names, f"Missing tool: {tool}"


@pytest.mark.integration
@pytest.mark.asyncio
class TestCortexQueryTools:
    """Test sanctuary_cortex query tools via direct SSE."""
    
    async def test_cortex_get_stats(self, cortex_cluster):
        """Test cortex stats via direct SSE."""
        result = await cortex_cluster.call_tool("cortex-get-stats", {})
        
        assert len(result.content) > 0
        assert "Total Memories" in result.content[0].text or "collections" in result.content[0].text
    
    async def test_cortex_query(self, cortex_cluster):
        """Test cortex query via direct SSE."""
        result = await cortex_cluster.call_tool("cortex-query", {
            "query": "Protocol 101",
            "max_results": 3
        })
        
        assert len(result.content) > 0
        # Just ensure we get a response, even if empty matches
        assert isinstance(result.content[0].text, str)
