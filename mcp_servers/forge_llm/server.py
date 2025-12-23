#============================================
# Path: mcp_servers/forge_llm/server.py
# Purpose: MCP Server for interacting with the fine-tuned Sanctuary model.
# Role: Library Server
# Used as: Standardized entry point for the Forge LLM MCP service.
# Calling example:
#   python3 -m mcp_servers.forge_llm.server
# LIST OF FUNCTIONS:
#   - check_sanctuary_model_status
#   - get_ops
#   - get_validator
#   - query_sanctuary_model
#============================================
from fastmcp import FastMCP
from .operations import ForgeOperations
from .validator import ForgeValidator, ValidationError
from .models import to_dict
import os
import json
import sys
from typing import Optional

# Initialize FastMCP with canonical domain name
mcp = FastMCP("project_sanctuary.system.forge")

# Start lazy instances
_forge_ops = None
_forge_validator = None
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")


#============================================
# Function: get_ops
# Purpose: Lazy initialization of ForgeOperations.
# Returns: ForgeOperations instance
#============================================
def get_ops() -> ForgeOperations:
    global _forge_ops
    if _forge_ops is None:
        _forge_ops = ForgeOperations(PROJECT_ROOT)
    return _forge_ops


#============================================
# Function: get_validator
# Purpose: Lazy initialization of ForgeValidator.
# Returns: ForgeValidator instance
#============================================
def get_validator() -> ForgeValidator:
    global _forge_validator
    if _forge_validator is None:
        _forge_validator = ForgeValidator(PROJECT_ROOT)
    return _forge_validator


#============================================
# Function: query_sanctuary_model
# Purpose: Query the fine-tuned Sanctuary model.
# Args:
#   prompt: Target prompt string
#   temperature: Sampling temperature
#   max_tokens: Recovery token limit
#   system_prompt: Optional system context
# Returns: JSON string with response
#============================================
@mcp.tool()
def query_sanctuary_model(
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    system_prompt: Optional[str] = None
) -> str:
    try:
        # Validate inputs
        validated = get_validator().validate_query_sanctuary_model(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt
        )
        
        # Query the model
        response = get_ops().query_sanctuary_model(
            prompt=validated["prompt"],
            temperature=validated["temperature"],
            max_tokens=validated["max_tokens"],
            system_prompt=validated["system_prompt"]
        )
        
        # Convert to dict and return as JSON
        result = to_dict(response)
        return json.dumps(result, indent=2)
        
    except ValidationError as e:
        return json.dumps({
            "status": "error",
            "error": f"Validation error: {str(e)}"
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e)
        }, indent=2)


#============================================
# Function: check_sanctuary_model_status
# Purpose: Check model availability and readiness.
# Returns: JSON string with status
#============================================
@mcp.tool()
def check_sanctuary_model_status() -> str:
    try:
        result = get_ops().check_model_availability()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e)
        }, indent=2)


if __name__ == "__main__":
    try:
        if not os.environ.get("SKIP_CONTAINER_CHECKS"):
            from mcp_servers.lib.container_manager import ensure_ollama_running
            # Ensure Ollama container is running
            print(f"Checking Ollama service...", file=sys.stderr)
            success, message = ensure_ollama_running(PROJECT_ROOT)
            if success:
                print(f"✓ {message}", file=sys.stderr)
            else:
                print(f"✗ {message}", file=sys.stderr)
                print("Model operations may fail without Ollama service", file=sys.stderr)
        else:
             print("Skipping container checks (SKIP_CONTAINER_CHECKS set)", file=sys.stderr)
    except ImportError:
        # If lib not available yet, proceed without check (or log warning)
        print("Warning: Could not import container_manager check", file=sys.stderr)
        pass

    mcp.run()
