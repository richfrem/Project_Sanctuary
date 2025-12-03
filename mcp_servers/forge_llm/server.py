"""
Forge MCP Server
Domain: project_sanctuary.system.forge

Provides MCP tools for interacting with the fine-tuned Sanctuary model.
"""
from fastmcp import FastMCP
from .operations import ForgeOperations
from .validator import ForgeValidator, ValidationError
from .models import to_dict
import os
import json
from typing import Optional

# Initialize FastMCP with canonical domain name
mcp = FastMCP("project_sanctuary.system.forge")

# Initialize operations and validator
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
forge_ops = ForgeOperations(PROJECT_ROOT)
forge_validator = ForgeValidator(PROJECT_ROOT)


@mcp.tool()
def query_sanctuary_model(
    prompt: str,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    system_prompt: Optional[str] = None
) -> str:
    """
    Query the fine-tuned Sanctuary model for specialized knowledge and decision-making.
    
    This tool enables LLM assistants to consult the custom-trained Sanctuary-Qwen2
    model for Project Sanctuary-specific knowledge, strategic insights, and
    protocol-aware responses.
    
    Args:
        prompt: The question or prompt to send to the Sanctuary model
        temperature: Sampling temperature (0.0-2.0, default: 0.7)
                    Lower = more focused, Higher = more creative
        max_tokens: Maximum tokens to generate (1-8192, default: 2048)
        system_prompt: Optional system prompt to set context
        
    Returns:
        JSON string with the model's response and metadata
        
    Example:
        query_sanctuary_model("What is the strategic priority for Q1 2025?")
        query_sanctuary_model(
            prompt="Explain Protocol 101",
            temperature=0.3,
            system_prompt="You are a Sanctuary protocol expert"
        )
    """
    try:
        # Validate inputs
        validated = forge_validator.validate_query_sanctuary_model(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt
        )
        
        # Query the model
        response = forge_ops.query_sanctuary_model(
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


@mcp.tool()
def check_sanctuary_model_status() -> str:
    """
    Check if the Sanctuary model is available and ready to use.
    
    Verifies that the fine-tuned Sanctuary-Qwen2 model is loaded in Ollama
    and ready for queries.
    
    Returns:
        JSON string with model availability status
        
    Example:
        check_sanctuary_model_status()
    """
    try:
        result = forge_ops.check_model_availability()
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "error": str(e)
        }, indent=2)


if __name__ == "__main__":
    mcp.run()
