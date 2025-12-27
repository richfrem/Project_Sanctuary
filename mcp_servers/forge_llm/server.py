#============================================
# Path: mcp_servers/forge_llm/server.py
# Purpose: MCP Server for interacting with the fine-tuned Sanctuary model.
# Role: Interface Layer
# Used as: Standardized entry point for the Forge LLM MCP service.
#============================================

import os
import json
import sys
from typing import Optional, Dict, Any
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

# Local/Library Imports
from mcp_servers.lib.env_helper import get_env_variable
from mcp_servers.lib.path_utils import find_project_root
from mcp_servers.lib.logging_utils import setup_mcp_logging
from .operations import ForgeOperations
from .validator import ForgeValidator, ValidationError
from .models import to_dict, ForgeQueryRequest

# 1. Initialize Logging
logger = setup_mcp_logging("project_sanctuary.system.forge")

# 2. Initialize FastMCP with Sanctuary Metadata
mcp = FastMCP(
    "project_sanctuary.system.forge",
    instructions="""
    Use this server to interact with the fine-tuned Sanctuary model (Codestral/Mistral).
    - Query the model for specialized coding assistance or sanctuary-aligned reasoning.
    - Check model status to verify Ollama availability.
    """
)

# 3. Global Instances (Lazy Init)
_forge_ops = None
_forge_validator = None
PROJECT_ROOT = get_env_variable("PROJECT_ROOT", required=False) or find_project_root()

def get_ops() -> ForgeOperations:
    global _forge_ops
    if _forge_ops is None:
        _forge_ops = ForgeOperations(PROJECT_ROOT)
    return _forge_ops

def get_validator() -> ForgeValidator:
    global _forge_validator
    if _forge_validator is None:
        _forge_validator = ForgeValidator(PROJECT_ROOT)
    return _forge_validator

#============================================
# Standardized Tool Implementations
#============================================

@mcp.tool()
async def query_sanctuary_model(request: ForgeQueryRequest) -> str:
    """
    Query the fine-tuned Sanctuary model for reasoning or code generation.
    """
    try:
        # 1. Domain Validation
        validated = get_validator().validate_query_sanctuary_model(
            prompt=request.prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            system_prompt=request.system_prompt
        )
        
        # 2. Delegate to Logic Layer
        response = get_ops().query_sanctuary_model(
            prompt=validated["prompt"],
            temperature=validated["temperature"],
            max_tokens=validated["max_tokens"],
            system_prompt=validated["system_prompt"]
        )
        
        # 3. Return Standardized Output
        return json.dumps(to_dict(response), indent=2)
        
    except ValidationError as e:
        logger.warning(f"Validation error in query_sanctuary_model: {e}")
        raise ToolError(f"Validation failed: {str(e)}")
    except Exception as e:
        logger.error(f"Error in query_sanctuary_model: {e}")
        raise ToolError(f"Query failed: {str(e)}")

@mcp.tool()
async def check_sanctuary_model_status() -> str:
    """
    Check the availability and operational readiness of the Sanctuary model.
    """
    try:
        result = get_ops().check_model_availability()
        return json.dumps(result, indent=2)
    except Exception as e:
        logger.error(f"Error in check_sanctuary_model_status: {e}")
        raise ToolError(f"Status check failed: {str(e)}")

#============================================
# Main Execution Entry Point
#============================================

if __name__ == "__main__":
    # Container / Service Pre-checks
    try:
        if not os.environ.get("SKIP_CONTAINER_CHECKS"):
            from mcp_servers.lib.container_manager import ensure_ollama_running
            success, message = ensure_ollama_running(PROJECT_ROOT)
            if not success:
                logger.warning(f"Ollama check failed: {message}")
    except Exception:
        pass

    # Dual-mode support:
    # 1. If PORT is set -> Run as SSE (Gateway Mode)
    # 2. If PORT is NOT set -> Run as Stdio (Local/CLI Mode)
    port_env = get_env_variable("PORT", required=False)
    transport = "sse" if port_env else "stdio"
    
    if transport == "sse":
        port = int(port_env) if port_env else 8007
        mcp.run(port=port, transport=transport)
    else:
        mcp.run(transport=transport)
