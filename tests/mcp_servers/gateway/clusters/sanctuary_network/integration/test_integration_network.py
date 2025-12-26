"""
Integration tests for sanctuary_network cluster.
Tests direct SSE communication (no Gateway).
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from integration_conftest import *


@pytest.mark.integration
class TestNetworkClusterHealth:
    """Test sanctuary_network cluster connectivity."""
    
    def test_cluster_health(self, network_cluster):
        """Verify cluster health endpoint."""
        assert network_cluster.health_check()
    
    def test_list_tools(self, network_cluster):
        """Verify network tools are exposed."""
        result = network_cluster.list_tools()
        assert result["success"]
        
        tools = result["tools"]
        tool_names = [t["name"] for t in tools]
        
        expected_tools = ["fetch-url", "check-site-status"]
        for tool in expected_tools:
            assert tool in tool_names, f"Missing tool: {tool}"


@pytest.mark.integration
class TestNetworkTools:
    """Test sanctuary_network tools via direct SSE."""
    
    def test_check_site_status(self, network_cluster):
        """Test checking site status via direct SSE."""
        result = network_cluster.call_tool("check-site-status", {"url": "https://www.google.com"})
        
        assert result["success"], f"Tool call failed: {result.get('error')}"
