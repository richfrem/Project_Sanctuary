"""
Integration tests for sanctuary_cortex cluster.
Tests direct SSE communication (no Gateway).
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from integration_conftest import *


@pytest.mark.integration
class TestCortexClusterHealth:
    """Test sanctuary_cortex cluster connectivity."""
    
    def test_cluster_health(self, cortex_cluster):
        """Verify cluster health endpoint."""
        assert cortex_cluster.health_check()
    
    def test_list_tools(self, cortex_cluster):
        """Verify cortex tools are exposed."""
        result = cortex_cluster.list_tools()
        assert result["success"]
        
        tools = result["tools"]
        tool_names = [t["name"] for t in tools]
        
        expected_tools = ["cortex-get-stats", "cortex-query"]
        for tool in expected_tools:
            assert tool in tool_names, f"Missing tool: {tool}"


@pytest.mark.integration
class TestCortexQueryTools:
    """Test sanctuary_cortex query tools via direct SSE."""
    
    def test_cortex_get_stats(self, cortex_cluster):
        """Test cortex stats via direct SSE."""
        result = cortex_cluster.call_tool("cortex-get-stats", {})
        
        assert result["success"], f"Tool call failed: {result.get('error')}"
    
    def test_cortex_query(self, cortex_cluster):
        """Test cortex query via direct SSE."""
        result = cortex_cluster.call_tool("cortex-query", {
            "query": "Protocol 101",
            "max_results": 3
        })
        
        assert result["success"], f"Tool call failed: {result.get('error')}"
