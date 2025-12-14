import pytest
from abc import ABC
from typing import Optional, Dict, Any
import asyncio

class BaseE2ETest(ABC):
    """
    Base class for E2E (End-to-End) MCP Tests.
    
    These tests sit at Layer 3 of the Testing Pyramid.
    They validate the FULL MCP client call lifecycle through the MCP protocol,
    requiring all 12 MCP servers to be running.
    
    Key Characteristics:
    - Tests via MCP protocol (not direct Python imports)
    - Requires the `mcp_servers` pytest fixture
    - Tests complete user scenarios
    - Slowest test layer (requires all servers)
    
    Usage:
        @pytest.mark.e2e
        class TestRAGCortexE2E(BaseE2ETest):
            async def test_query_via_mcp(self, mcp_servers):
                result = await self.call_mcp_tool(
                    "cortex_query",
                    {"query": "What is Protocol 101?", "max_results": 3}
                )
                assert result["status"] == "success"
    
    Note: The `mcp_servers` fixture is defined in tests/conftest.py and
    automatically starts/stops all MCP servers for the test session.
    """
    
    async def call_mcp_tool(
        self, 
        tool_name: str, 
        arguments: Dict[str, Any],
        timeout: int = 30
    ) -> Dict[str, Any]:
        """
        Call an MCP tool through the MCP client.
        
        Args:
            tool_name: Name of the MCP tool to call
            arguments: Tool arguments as a dictionary
            timeout: Timeout in seconds (default: 30)
            
        Returns:
            Tool result as a dictionary
            
        Raises:
            TimeoutError: If the call exceeds the timeout
            RuntimeError: If the MCP call fails
        """
        # TODO: Implement actual MCP client call
        # This will be implemented when we have the MCP client library integrated
        # For now, this is a placeholder structure
        raise NotImplementedError(
            "MCP client integration not yet implemented. "
            "This will use the MCP SDK to make actual protocol calls."
        )
    
    def assert_mcp_success(self, result: Dict[str, Any], message: str = ""):
        """
        Assert that an MCP tool call was successful.
        
        Args:
            result: The result dictionary from an MCP tool call
            message: Optional custom assertion message
        """
        assert "status" in result, f"MCP result missing 'status' field. {message}"
        assert result["status"] == "success", (
            f"MCP call failed: {result.get('error', 'Unknown error')}. {message}"
        )
