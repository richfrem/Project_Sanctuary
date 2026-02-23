"""
Integration tests for sanctuary_git cluster.
Tests direct SSE communication via MCP SDK.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from integration_conftest import *


@pytest.mark.integration
@pytest.mark.asyncio
class TestGitClusterHealth:
    """Test sanctuary_git cluster connectivity."""
    
    async def test_cluster_health(self, git_cluster):
        """Verify cluster health endpoint."""
        assert await git_cluster.health_check()
    
    async def test_list_tools(self, git_cluster):
        """Verify git tools are exposed."""
        result = await git_cluster.list_tools()
        tool_names = [t.name for t in result.tools]
        
        expected_tools = ["git-get-status", "git-log", "git-diff"]
        for tool in expected_tools:
            assert tool in tool_names, f"Missing tool: {tool}"


@pytest.mark.integration
@pytest.mark.asyncio
class TestGitReadTools:
    """Test sanctuary_git read tools via direct SSE."""
    
    async def test_git_get_status(self, git_cluster):
        """Test git status via direct SSE."""
        result = await git_cluster.call_tool("git-get-status", {})
        
        assert len(result.content) > 0
        # Should return text status
        assert isinstance(result.content[0].text, str)
    
    async def test_git_log(self, git_cluster):
        """Test git log via direct SSE."""
        result = await git_cluster.call_tool("git-log", {"max_count": 5})
        
        assert len(result.content) > 0
        assert "commit" in result.content[0].text or "Date:" in result.content[0].text
