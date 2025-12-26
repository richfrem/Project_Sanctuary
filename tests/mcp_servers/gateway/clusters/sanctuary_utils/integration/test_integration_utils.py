"""
Integration tests for sanctuary_utils cluster.
Tests direct SSE communication (no Gateway).
"""
import pytest
import sys
from pathlib import Path

# Import shared fixtures
sys.path.insert(0, str(Path(__file__).parents[2]))
from integration_conftest import *


@pytest.mark.integration
class TestUtilsClusterHealth:
    """Test sanctuary_utils cluster connectivity."""
    
    def test_cluster_health(self, utils_cluster):
        """Verify cluster health endpoint."""
        assert utils_cluster.health_check(), "Cluster health check failed"
    
    def test_list_tools(self, utils_cluster):
        """Verify cluster exposes expected tools."""
        result = utils_cluster.list_tools()
        assert result["success"], f"Failed to list tools: {result.get('error')}"
        
        tools = result["tools"]
        tool_names = [t["name"] for t in tools]
        
        expected_tools = [
            "time-get-current-time",
            "calculator-add",
            "uuid-generate-uuid4",
            "string-to-upper",
        ]
        
        for tool in expected_tools:
            assert tool in tool_names, f"Missing tool: {tool}"


@pytest.mark.integration
class TestUtilsTimeTools:
    """Test sanctuary_utils time tools via direct SSE."""
    
    def test_time_get_current_time(self, utils_cluster):
        """Test time-get-current-time via direct SSE."""
        result = utils_cluster.call_tool("time-get-current-time", {"timezone_name": "UTC"})
        
        assert result["success"], f"Tool call failed: {result.get('error')}"
        content = result["result"]["content"]
        assert len(content) > 0, "No content returned"
        assert "text" in content[0], "Missing text field"


@pytest.mark.integration
class TestUtilsCalculatorTools:
    """Test sanctuary_utils calculator tools via direct SSE."""
    
    def test_calculator_add(self, utils_cluster):
        """Test calculator-add via direct SSE."""
        result = utils_cluster.call_tool("calculator-add", {"a": 5, "b": 3})
        
        assert result["success"], f"Tool call failed: {result.get('error')}"
        content = result["result"]["content"]
        assert "8" in content[0]["text"], "Incorrect calculation"
    
    def test_calculator_subtract(self, utils_cluster):
        """Test calculator-subtract via direct SSE."""
        result = utils_cluster.call_tool("calculator-subtract", {"a": 10, "b": 4})
        
        assert result["success"], f"Tool call failed: {result.get('error')}"
        content = result["result"]["content"]
        assert "6" in content[0]["text"], "Incorrect calculation"
    
    def test_calculator_multiply(self, utils_cluster):
        """Test calculator-multiply via direct SSE."""
        result = utils_cluster.call_tool("calculator-multiply", {"a": 7, "b": 6})
        
        assert result["success"], f"Tool call failed: {result.get('error')}"
        content = result["result"]["content"]
        assert "42" in content[0]["text"], "Incorrect calculation"


@pytest.mark.integration
class TestUtilsStringTools:
    """Test sanctuary_utils string tools via direct SSE."""
    
    def test_string_to_upper(self, utils_cluster):
        """Test string-to-upper via direct SSE."""
        result = utils_cluster.call_tool("string-to-upper", {"text": "hello world"})
        
        assert result["success"], f"Tool call failed: {result.get('error')}"
        content = result["result"]["content"]
        assert "HELLO WORLD" in content[0]["text"], "String not uppercased"
    
    def test_string_to_lower(self, utils_cluster):
        """Test string-to-lower via direct SSE."""
        result = utils_cluster.call_tool("string-to-lower", {"text": "HELLO WORLD"})
        
        assert result["success"], f"Tool call failed: {result.get('error')}"
        content = result["result"]["content"]
        assert "hello world" in content[0]["text"], "String not lowercased"


@pytest.mark.integration
class TestUtilsUUIDTools:
    """Test sanctuary_utils UUID tools via direct SSE."""
    
    def test_uuid_generate_uuid4(self, utils_cluster):
        """Test uuid-generate-uuid4 via direct SSE."""
        result = utils_cluster.call_tool("uuid-generate-uuid4", {})
        
        assert result["success"], f"Tool call failed: {result.get('error')}"
        content = result["result"]["content"]
        # UUID4 format: 8-4-4-4-12
        assert "-" in content[0]["text"], "Invalid UUID format"


@pytest.mark.integration
class TestUtilsErrorHandling:
    """Test error handling via direct SSE."""
    
    def test_invalid_tool_name(self, utils_cluster):
        """Verify proper error handling for invalid tool."""
        result = utils_cluster.call_tool("nonexistent-tool", {})
        
        assert not result["success"], "Should fail for invalid tool"
        assert "error" in result, "Should return error message"
