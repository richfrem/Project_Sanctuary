
import os
import sys
import logging
from pathlib import Path
from mcp.server.fastmcp import FastMCP, Context
import mcp.types as types

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from mcp_servers.lib.logging_utils import setup_mcp_logging
from mcp_servers.lib.env_helper import get_env_variable
from mcp_servers.evolution.operations import EvolutionOperations
from mcp_servers.lib.sse_adaptor import SSEServer

# Setup logging
logger = setup_mcp_logging("evolution")

# Initialize Operations
ops = EvolutionOperations(project_root)

# Initialize FastMCP
mcp = FastMCP("evolution")

#=============================================================================
# TOOLS (Protocol 131 Self-Improvement)
#=============================================================================

@mcp.tool(name="measure_fitness")
def measure_fitness(content: str) -> dict:
    """
    Calculates evolutionary fitness metrics (Depth, Scope) for a given text content.
    Used for Protocol 131 Map-Elites placement.
    """
    return ops.calculate_fitness(content)

@mcp.tool(name="evaluate_depth")
def evaluate_depth(content: str) -> float:
    """
    Calculates the 'Depth' score (0.0-5.0) for evolutionary selection.
    """
    return ops.measure_depth(content)

@mcp.tool(name="evaluate_scope")
def evaluate_scope(content: str) -> float:
    """
    Calculates the 'Scope' score (0.0-5.0) for evolutionary selection.
    """
    return ops.measure_scope(content)

#=============================================================================
# MAIN
#=============================================================================
if __name__ == "__main__":
    transport = get_env_variable("MCP_TRANSPORT", required=False) or "stdio"
    port = int(get_env_variable("PORT", required=False) or "8002") # Different default port

    logger.info(f"Starting Evolution MCP Server (Transport: {transport})...")

    if transport.lower() == "sse":
        sse = SSEServer(mcp, host="0.0.0.0", port=port)
        sse.start()
    else:
        mcp.run()
