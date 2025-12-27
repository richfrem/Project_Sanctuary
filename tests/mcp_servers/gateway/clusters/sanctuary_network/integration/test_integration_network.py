"""
Integration tests for sanctuary_network cluster.
Tests direct SSE communication via MCP SDK.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from integration_conftest import *


@pytest.mark.integration
@pytest.mark.asyncio
class TestNetworkClusterHealth:
    """Test sanctuary_network cluster connectivity."""
    
    async def test_cluster_health(self, network_cluster):
        """Verify cluster health endpoint."""
        assert await network_cluster.health_check()
    
    async def test_list_tools(self, network_cluster):
        """Verify network tools are exposed."""
        result = await network_cluster.list_tools()
        tool_names = [t.name for t in result.tools]
        
        expected_tools = ["fetch-url", "check-site-status"]
        for tool in expected_tools:
            assert tool in tool_names, f"Missing tool: {tool}"


@pytest.mark.integration
@pytest.mark.asyncio
class TestNetworkTools:
    """Test sanctuary_network tools via direct SSE."""
    
    async def test_check_site_status(self, network_cluster):
        """Test checking site status via direct SSE."""
        # Using a reliable site
        result = await network_cluster.call_tool("check-site-status", {"url": "https://www.google.com"})
        
        assert len(result.content) > 0
        # Should contain status code or status text
        assert "200" in result.content[0].text or "OK" in result.content[0].text
