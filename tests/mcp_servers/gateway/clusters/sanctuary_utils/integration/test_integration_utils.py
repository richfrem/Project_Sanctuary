"""
Integration tests for sanctuary_utils cluster.
Tests direct SSE communication via MCP SDK.
"""
import pytest
import sys
from pathlib import Path

# Import shared fixtures
sys.path.insert(0, str(Path(__file__).parents[2]))
from integration_conftest import *


@pytest.mark.integration
@pytest.mark.asyncio
class TestUtilsClusterHealth:
    """Test sanctuary_utils cluster connectivity."""
    
    async def test_cluster_health(self, utils_cluster):
        """Verify cluster health endpoint."""
        assert await utils_cluster.health_check(), "Cluster health check failed"
    
    async def test_list_tools(self, utils_cluster):
        """Verify cluster exposes expected tools."""
        result = await utils_cluster.list_tools()
        # MCP SDK returns ListToolsResult
        
        tool_names = [t.name for t in result.tools]
        
        expected_tools = [
            "time-get-current-time",
            "calculator-add",
            "uuid-generate-uuid4",
            "string-to-upper",
        ]
        
        for tool in expected_tools:
            assert tool in tool_names, f"Missing tool: {tool}"


@pytest.mark.integration
@pytest.mark.asyncio
class TestUtilsTimeTools:
    """Test sanctuary_utils time tools via direct SSE."""
    
    async def test_time_get_current_time(self, utils_cluster):
        """Test time-get-current-time via direct SSE."""
        # SDK returns CallToolResult
        result = await utils_cluster.call_tool("time-get-current-time", {"timezone_name": "UTC"})
        
        # Access via object attributes, not dict keys if using SDK objects
        # CallToolResult(content=[TextContent(type='text', text='...')])
        assert len(result.content) > 0, "No content returned"
        assert result.content[0].type == 'text', "Expected text content"
        assert len(result.content[0].text) > 0, "Missing text content"


@pytest.mark.integration
@pytest.mark.asyncio
class TestUtilsCalculatorTools:
    """Test sanctuary_utils calculator tools via direct SSE."""
    
    async def test_calculator_add(self, utils_cluster):
        """Test calculator-add via direct SSE."""
        result = await utils_cluster.call_tool("calculator-add", {"a": 5, "b": 3})
        
        assert "8" in result.content[0].text, "Incorrect calculation"
    
    async def test_calculator_subtract(self, utils_cluster):
        """Test calculator-subtract via direct SSE."""
        result = await utils_cluster.call_tool("calculator-subtract", {"a": 10, "b": 4})
        
        assert "6" in result.content[0].text, "Incorrect calculation"
    
    async def test_calculator_multiply(self, utils_cluster):
        """Test calculator-multiply via direct SSE."""
        result = await utils_cluster.call_tool("calculator-multiply", {"a": 7, "b": 6})
        
        assert "42" in result.content[0].text, "Incorrect calculation"


@pytest.mark.integration
@pytest.mark.asyncio
class TestUtilsStringTools:
    """Test sanctuary_utils string tools via direct SSE."""
    
    async def test_string_to_upper(self, utils_cluster):
        """Test string-to-upper via direct SSE."""
        result = await utils_cluster.call_tool("string-to-upper", {"text": "hello world"})
        
        assert "HELLO WORLD" in result.content[0].text, "String not uppercased"
    
    async def test_string_to_lower(self, utils_cluster):
        """Test string-to-lower via direct SSE."""
        result = await utils_cluster.call_tool("string-to-lower", {"text": "HELLO WORLD"})
        
        assert "hello world" in result.content[0].text, "String not lowercased"


@pytest.mark.integration
@pytest.mark.asyncio
class TestUtilsUUIDTools:
    """Test sanctuary_utils UUID tools via direct SSE."""
    
    async def test_uuid_generate_uuid4(self, utils_cluster):
        """Test uuid-generate-uuid4 via direct SSE."""
        result = await utils_cluster.call_tool("uuid-generate-uuid4", {})
        
        assert "-" in result.content[0].text, "Invalid UUID format"


@pytest.mark.integration
@pytest.mark.asyncio
class TestUtilsErrorHandling:
    """Test error handling via direct SSE."""
    
    async def test_invalid_tool_name(self, utils_cluster):
        """Verify proper error handling for invalid tool."""
        # The SDK likely raises an exception for method not found
        with pytest.raises(Exception): # Catch generic exception for now, ideally specific McpError
            await utils_cluster.call_tool("nonexistent-tool", {})
