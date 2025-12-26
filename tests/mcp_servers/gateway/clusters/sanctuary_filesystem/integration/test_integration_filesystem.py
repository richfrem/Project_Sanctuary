"""
Integration tests for sanctuary_filesystem cluster.
Tests direct SSE communication via MCP SDK.
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from integration_conftest import *


@pytest.mark.integration
@pytest.mark.asyncio
class TestFilesystemClusterHealth:
    """Test sanctuary_filesystem cluster connectivity."""
    
    async def test_cluster_health(self, filesystem_cluster):
        """Verify cluster health endpoint."""
        assert await filesystem_cluster.health_check()
    
    async def test_list_tools(self, filesystem_cluster):
        """Verify filesystem tools are exposed."""
        result = await filesystem_cluster.list_tools()
        tool_names = [t.name for t in result.tools]
        
        expected_tools = ["code-read", "code-write", "code-list-files"]
        for tool in expected_tools:
            assert tool in tool_names, f"Missing tool: {tool}"


@pytest.mark.integration
@pytest.mark.asyncio
class TestFilesystemReadTools:
    """Test sanctuary_filesystem read tools via direct SSE."""
    
    async def test_code_read(self, filesystem_cluster):
        """Test reading a file via direct SSE."""
        # Read the README.md in the container
        result = await filesystem_cluster.call_tool("code-read", {"path": "/app/README.md"})
        
        assert len(result.content) > 0
        assert "Sanctuary" in result.content[0].text or "Project" in result.content[0].text
    
    async def test_code_list_files(self, filesystem_cluster):
        """Test listing files via direct SSE."""
        result = await filesystem_cluster.call_tool("code-list-files", {
            "path": "/app",
            "pattern": "*.md",
            "recursive": False
        })
        
        assert len(result.content) > 0
        assert "README.md" in result.content[0].text
