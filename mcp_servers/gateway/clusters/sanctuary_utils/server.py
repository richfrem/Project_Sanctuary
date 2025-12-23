
"""
sanctuary_utils MCP Server
Fleet of 7 - Container #1: Low-risk utility tools

Refactored to use generic SSEServer for Gateway integration (202 Accepted + Async SSE).
"""
import sys
import os

# Ensure we can import from shared lib
# Add project root to path if needed (for local dev)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import shared SSE Server
try:
    from mcp_servers.lib.sse_adaptor import SSEServer
except ImportError:
    # Fallback for container structure where lib might be adjacent
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from lib.sse_adaptor import SSEServer

# Import tools
# Try generic relative import first, then absolute
try:
    from .tools import time_tool
    from .tools import calculator_tool
    from .tools import uuid_tool
    from .tools import string_tool
except ImportError:
    from tools import time_tool
    from tools import calculator_tool
    from tools import uuid_tool
    from tools import string_tool

# Initialize Server
server = SSEServer("sanctuary_utils")

# Tool Registry
# We manually register generic tools
# Time
server.register_tool("time.get_current_time", time_tool.get_current_time, time_tool.TOOL_MANIFEST)
server.register_tool("time.get_timezone_info", time_tool.get_timezone_info, time_tool.TOOL_MANIFEST)

# Calculator
server.register_tool("calculator.calculate", calculator_tool.calculate, calculator_tool.TOOL_MANIFEST)
server.register_tool("calculator.add", calculator_tool.add, calculator_tool.TOOL_MANIFEST)
server.register_tool("calculator.subtract", calculator_tool.subtract, calculator_tool.TOOL_MANIFEST)
server.register_tool("calculator.multiply", calculator_tool.multiply, calculator_tool.TOOL_MANIFEST)
server.register_tool("calculator.divide", calculator_tool.divide, calculator_tool.TOOL_MANIFEST)

# UUID
server.register_tool("uuid.generate_uuid4", uuid_tool.generate_uuid4, uuid_tool.TOOL_MANIFEST)
server.register_tool("uuid.generate_uuid1", uuid_tool.generate_uuid1, uuid_tool.TOOL_MANIFEST)
server.register_tool("uuid.validate_uuid", uuid_tool.validate_uuid, uuid_tool.TOOL_MANIFEST)

# String
server.register_tool("string.to_upper", string_tool.to_upper, string_tool.TOOL_MANIFEST)
server.register_tool("string.to_lower", string_tool.to_lower, string_tool.TOOL_MANIFEST)
server.register_tool("string.trim", string_tool.trim, string_tool.TOOL_MANIFEST)
server.register_tool("string.reverse", string_tool.reverse, string_tool.TOOL_MANIFEST)
server.register_tool("string.word_count", string_tool.word_count, string_tool.TOOL_MANIFEST)
server.register_tool("string.replace", string_tool.replace, string_tool.TOOL_MANIFEST)

# Meta Tools (Capabilities)
from mcp_servers.lib.capabilities_utils import get_gateway_capabilities
async def gateway_get_capabilities() -> str:
    """Returns a high-level overview of available MCP servers and their primary functions."""
    import json
    # Use the PROJECT_ROOT defined in the server's scope
    res = get_gateway_capabilities(project_root)
    return json.dumps(res, indent=2)

server.register_tool("gateway_get_capabilities", gateway_get_capabilities, {
    "type": "object",
    "properties": {}
})

# Expose app for uvicorn
app = server.app

if __name__ == "__main__":
    # Dual-mode support:
    # 1. If PORT is set -> Run as SSE (Gateway Mode)
    # 2. If PORT is NOT set -> Run as Stdio (Legacy Mode)
    import os
    port_env = os.getenv("PORT")
    transport = "sse" if port_env else "stdio"
    port = int(port_env) if port_env else 8100
    
    server.run(port=port, transport=transport)
