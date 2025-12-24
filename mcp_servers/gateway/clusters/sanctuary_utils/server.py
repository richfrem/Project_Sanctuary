#============================================
# mcp_servers/gateway/clusters/sanctuary_utils/server.py
# Purpose: Sanctuary Utils Cluster - Dual-Transport Entry Point
# Role: Interface Layer (Aggregator Node)
# Status: ADR-066 v1.3 Compliant (SSEServer for Gateway, FastMCP for STDIO)
# Used by: Gateway Fleet (SSE) and Claude Desktop (STDIO)
#============================================
# Transport Selection (per ADR-066 v1.3):
#   - MCP_TRANSPORT=sse  -> Uses SSEServer (Gateway-compatible)
#   - MCP_TRANSPORT=stdio -> Uses FastMCP (local development)
#   - Default: stdio (safe for local)
#============================================

import os
import sys
import json
import logging
from typing import Optional

# Local/Library Imports
from mcp_servers.lib.env_helper import get_env_variable
from mcp_servers.lib.path_utils import find_project_root
from mcp_servers.lib.logging_utils import setup_mcp_logging
from mcp_servers.lib.capabilities_utils import get_gateway_capabilities

# Import Tool Logic Modules (shared by both transports)
from mcp_servers.gateway.clusters.sanctuary_utils.tools import time_tool
from mcp_servers.gateway.clusters.sanctuary_utils.tools import calculator_tool
from mcp_servers.gateway.clusters.sanctuary_utils.tools import uuid_tool
from mcp_servers.gateway.clusters.sanctuary_utils.tools import string_tool

# Setup Logging
logger = setup_mcp_logging("project_sanctuary.sanctuary_utils")

# Configuration
PROJECT_ROOT = get_env_variable("PROJECT_ROOT", required=False) or find_project_root()

#============================================
# Tool Schema Definitions (for SSEServer registration)
#============================================
TIME_CURRENT_SCHEMA = {
    "type": "object",
    "properties": {
        "timezone_name": {"type": "string", "description": "Timezone name (default: UTC)"}
    }
}

CALC_EXPRESSION_SCHEMA = {
    "type": "object",
    "properties": {
        "expression": {"type": "string", "description": "Math expression to evaluate"}
    },
    "required": ["expression"]
}

CALC_BINARY_SCHEMA = {
    "type": "object",
    "properties": {
        "a": {"type": "number", "description": "First number"},
        "b": {"type": "number", "description": "Second number"}
    },
    "required": ["a", "b"]
}

UUID_VALIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "uuid_string": {"type": "string", "description": "UUID string to validate"}
    },
    "required": ["uuid_string"]
}

STRING_SINGLE_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "description": "Text to process"}
    },
    "required": ["text"]
}

STRING_REPLACE_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string", "description": "Original text"},
        "old": {"type": "string", "description": "Substring to replace"},
        "new": {"type": "string", "description": "Replacement substring"}
    },
    "required": ["text", "old", "new"]
}

EMPTY_SCHEMA = {"type": "object", "properties": {}}


#============================================
# SSE Transport Implementation (Gateway Mode)
#============================================
def run_sse_server(port: int):
    """Run using SSEServer for Gateway compatibility (ADR-066 v1.3)."""
    from mcp_servers.lib.sse_adaptor import SSEServer
    
    server = SSEServer("sanctuary_utils", version="1.0.0")
    
    # =============================================================================
    # TIME TOOLS
    # =============================================================================
    server.register_tool("time_get_current_time", time_tool.get_current_time, TIME_CURRENT_SCHEMA)
    server.register_tool("time_get_timezone_info", time_tool.get_timezone_info, EMPTY_SCHEMA)
    
    # =============================================================================
    # CALCULATOR TOOLS
    # =============================================================================
    server.register_tool("calculator_calculate", calculator_tool.calculate, CALC_EXPRESSION_SCHEMA)
    server.register_tool("calculator_add", calculator_tool.add, CALC_BINARY_SCHEMA)
    server.register_tool("calculator_subtract", calculator_tool.subtract, CALC_BINARY_SCHEMA)
    server.register_tool("calculator_multiply", calculator_tool.multiply, CALC_BINARY_SCHEMA)
    server.register_tool("calculator_divide", calculator_tool.divide, CALC_BINARY_SCHEMA)
    
    # =============================================================================
    # UUID TOOLS
    # =============================================================================
    server.register_tool("uuid_generate_uuid4", uuid_tool.generate_uuid4, EMPTY_SCHEMA)
    server.register_tool("uuid_generate_uuid1", uuid_tool.generate_uuid1, EMPTY_SCHEMA)
    server.register_tool("uuid_validate_uuid", uuid_tool.validate_uuid, UUID_VALIDATE_SCHEMA)
    
    # =============================================================================
    # STRING TOOLS
    # =============================================================================
    server.register_tool("string_to_upper", string_tool.to_upper, STRING_SINGLE_SCHEMA)
    server.register_tool("string_to_lower", string_tool.to_lower, STRING_SINGLE_SCHEMA)
    server.register_tool("string_trim", string_tool.trim, STRING_SINGLE_SCHEMA)
    server.register_tool("string_reverse", string_tool.reverse, STRING_SINGLE_SCHEMA)
    server.register_tool("string_word_count", string_tool.word_count, STRING_SINGLE_SCHEMA)
    server.register_tool("string_replace", string_tool.replace, STRING_REPLACE_SCHEMA)
    
    # =============================================================================
    # META TOOLS
    # =============================================================================
    async def gateway_get_capabilities_handler():
        """Returns a high-level overview of available MCP servers."""
        return get_gateway_capabilities(PROJECT_ROOT)
    
    server.register_tool("gateway_get_capabilities", gateway_get_capabilities_handler, EMPTY_SCHEMA)
    
    logger.info(f"Starting SSEServer on port {port} (Gateway Mode)")
    server.run(port=port, transport="sse")


#============================================
# STDIO Transport Implementation (Local Mode)
#============================================
def run_stdio_server():
    """Run using FastMCP for local development (Claude Desktop)."""
    from fastmcp import FastMCP
    from fastmcp.exceptions import ToolError
    from mcp_servers.gateway.clusters.sanctuary_utils.models import (
        TimeCurrentRequest, CalcExpressionRequest, CalcBinaryRequest,
        UUIDValidateRequest, StringSingleRequest, StringReplaceRequest
    )
    
    mcp = FastMCP(
        "sanctuary_utils",
        instructions="""
        Sanctuary Utility Cluster.
        - provides generic services: Time, Calculator, UUID, and String manipulation.
        - provides meta-information about Gateway capabilities.
        """
    )
    
    # =============================================================================
    # TIME TOOLS
    # =============================================================================
    @mcp.tool()
    async def time_get_current_time(request: Optional[TimeCurrentRequest] = None) -> dict:
        """Get the current time in the specified timezone."""
        request = request or TimeCurrentRequest()
        return time_tool.get_current_time(request.timezone_name)
    
    @mcp.tool()
    async def time_get_timezone_info() -> dict:
        """Get information about available timezones."""
        return time_tool.get_timezone_info()
    
    # =============================================================================
    # CALCULATOR TOOLS
    # =============================================================================
    @mcp.tool()
    async def calculator_calculate(request: CalcExpressionRequest) -> dict:
        """Evaluate a mathematical expression safely."""
        return calculator_tool.calculate(request.expression)
    
    @mcp.tool()
    async def calculator_add(request: CalcBinaryRequest) -> dict:
        """Add two numbers."""
        return calculator_tool.add(request.a, request.b)
    
    @mcp.tool()
    async def calculator_subtract(request: CalcBinaryRequest) -> dict:
        """Subtract b from a."""
        return calculator_tool.subtract(request.a, request.b)
    
    @mcp.tool()
    async def calculator_multiply(request: CalcBinaryRequest) -> dict:
        """Multiply two numbers."""
        return calculator_tool.multiply(request.a, request.b)
    
    @mcp.tool()
    async def calculator_divide(request: CalcBinaryRequest) -> dict:
        """Divide a by b."""
        return calculator_tool.divide(request.a, request.b)
    
    # =============================================================================
    # UUID TOOLS
    # =============================================================================
    @mcp.tool()
    async def uuid_generate_uuid4() -> dict:
        """Generate a random UUID (version 4)."""
        return uuid_tool.generate_uuid4()
    
    @mcp.tool()
    async def uuid_generate_uuid1() -> dict:
        """Generate a UUID based on host ID and current time (version 1)."""
        return uuid_tool.generate_uuid1()
    
    @mcp.tool()
    async def uuid_validate_uuid(request: UUIDValidateRequest) -> dict:
        """Validate if a string is a valid UUID."""
        return uuid_tool.validate_uuid(request.uuid_string)
    
    # =============================================================================
    # STRING TOOLS
    # =============================================================================
    @mcp.tool()
    async def string_to_upper(request: StringSingleRequest) -> dict:
        """Convert text to uppercase."""
        return string_tool.to_upper(request.text)
    
    @mcp.tool()
    async def string_to_lower(request: StringSingleRequest) -> dict:
        """Convert text to lowercase."""
        return string_tool.to_lower(request.text)
    
    @mcp.tool()
    async def string_trim(request: StringSingleRequest) -> dict:
        """Remove leading and trailing whitespace."""
        return string_tool.trim(request.text)
    
    @mcp.tool()
    async def string_reverse(request: StringSingleRequest) -> dict:
        """Reverse a string."""
        return string_tool.reverse(request.text)
    
    @mcp.tool()
    async def string_word_count(request: StringSingleRequest) -> dict:
        """Count words in text."""
        return string_tool.word_count(request.text)
    
    @mcp.tool()
    async def string_replace(request: StringReplaceRequest) -> dict:
        """Replace occurrences of old with new in text."""
        return string_tool.replace(request.text, request.old, request.new)
    
    # =============================================================================
    # META TOOLS
    # =============================================================================
    @mcp.tool()
    async def gateway_get_capabilities() -> str:
        """Returns a high-level overview of available MCP servers."""
        try:
            res = get_gateway_capabilities(PROJECT_ROOT)
            return json.dumps(res, indent=2)
        except Exception as e:
            logger.error(f"Error in gateway_get_capabilities: {e}")
            raise ToolError(f"Failed to get capabilities: {str(e)}")
    
    logger.info("Starting FastMCP server (STDIO Mode)")
    mcp.run(transport="stdio")


#============================================
# Main Execution Entry Point (ADR-066 v1.3 Canonical Selector)
#============================================
def run_server():
    """
    Start the server with transport based on MCP_TRANSPORT env variable.
    
    Per ADR-066 v1.3:
    - MCP_TRANSPORT=sse -> SSEServer (Gateway compatible)
    - MCP_TRANSPORT=stdio -> FastMCP (local development)
    - Default: stdio
    """
    MCP_TRANSPORT = os.getenv("MCP_TRANSPORT", "stdio").lower()
    
    if MCP_TRANSPORT not in {"stdio", "sse"}:
        logger.error(f"Invalid MCP_TRANSPORT: {MCP_TRANSPORT}. Must be 'stdio' or 'sse'.")
        sys.exit(1)
    
    if MCP_TRANSPORT == "sse":
        port = int(os.getenv("PORT", 8000))
        run_sse_server(port)
    else:
        run_stdio_server()


if __name__ == "__main__":
    run_server()
