"""
Integration tests for sanctuary_filesystem cluster.
Tests direct SSE communication (no Gateway).
"""
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))
from integration_conftest import *


@pytest.mark.integration
class TestFilesystemClusterHealth:
    """Test sanctuary_filesystem cluster connectivity."""
    
    def test_cluster_health(self, filesystem_cluster):
        """Verify cluster health endpoint."""
        assert filesystem_cluster.health_check()
    
    def test_list_tools(self, filesystem_cluster):
        """Verify filesystem tools are exposed."""
        result = filesystem_cluster.list_tools()
        assert result["success"]
        
        tools = result["tools"]
        tool_names = [t["name"] for t in tools]
        
        expected_tools = ["code-read", "code-write", "code-list-files"]
        for tool in expected_tools:
            assert tool in tool_names, f"Missing tool: {tool}"


@pytest.mark.integration
class TestFilesystemReadTools:
    """Test sanctuary_filesystem read tools via direct SSE."""
    
    def test_code_read(self, filesystem_cluster):
        """Test reading a file via direct SSE."""
        result = filesystem_cluster.call_tool("code-read", {"path": "/app/README.md"})
        
        assert result["success"], f"Tool call failed: {result.get('error')}"
        content = result["result"]["content"]
        assert len(content) > 0
    
    def test_code_list_files(self, filesystem_cluster):
        """Test listing files via direct SSE."""
        result = filesystem_cluster.call_tool("code-list-files", {
            "path": "/app",
            "pattern": "*.md",
            "recursive": False
        })
        
        assert result["success"], f"Tool call failed: {result.get('error')}"
