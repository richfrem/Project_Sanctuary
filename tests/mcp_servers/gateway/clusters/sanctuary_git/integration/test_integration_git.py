"""
Integration tests for sanctuary_git cluster.
Tests direct SSE communication (no Gateway).
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from integration_conftest import *


@pytest.mark.integration
class TestGitClusterHealth:
    """Test sanctuary_git cluster connectivity."""
    
    def test_cluster_health(self, git_cluster):
        """Verify cluster health endpoint."""
        assert git_cluster.health_check()
    
    def test_list_tools(self, git_cluster):
        """Verify git tools are exposed."""
        result = git_cluster.list_tools()
        assert result["success"]
        
        tools = result["tools"]
        tool_names = [t["name"] for t in tools]
        
        expected_tools = ["git-get-status", "git-log", "git-diff"]
        for tool in expected_tools:
            assert tool in tool_names, f"Missing tool: {tool}"


@pytest.mark.integration
class TestGitReadTools:
    """Test sanctuary_git read tools via direct SSE."""
    
    def test_git_get_status(self, git_cluster):
        """Test git status via direct SSE."""
        result = git_cluster.call_tool("git-get-status", {})
        
        assert result["success"], f"Tool call failed: {result.get('error')}"
        content = result["result"]["content"]
        assert len(content) > 0
    
    def test_git_log(self, git_cluster):
        """Test git log via direct SSE."""
        result = git_cluster.call_tool("git-log", {"max_count": 5})
        
        assert result["success"], f"Tool call failed: {result.get('error')}"
