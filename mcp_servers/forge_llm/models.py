#============================================
# Path: mcp_servers/forge_llm/models.py
# Purpose: Data models for Forge MCP operations.
# Role: Library Models
# Used as: Common data structures for the Forge LLM library.
# Calling example:
#   from mcp_servers.forge_llm.models import ModelQueryResponse
# LIST OF CLASSES/FUNCTIONS:
#   - ModelQueryResponse
#   - to_dict
#============================================
from dataclasses import dataclass
from typing import Optional, Dict, Any


#============================================
# Class: ModelQueryResponse
# Purpose: Response from querying the Sanctuary model.
#============================================
@dataclass
class ModelQueryResponse:
    model: str
    response: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    temperature: Optional[float] = None
    status: str = "success"
    error: Optional[str] = None


#============================================
# Function: to_dict
# Purpose: Convert dataclass to dictionary.
# Args:
#   obj: The object to convert
# Returns: Dictionary representation
#============================================
def to_dict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, '__dataclass_fields__'):
        return {k: v for k, v in obj.__dict__.items() if v is not None}
    return obj

#============================================
# FastMCP Request Models
#============================================
from pydantic import BaseModel, Field

class ForgeQueryRequest(BaseModel):
    prompt: str = Field(..., description="The textual prompt for the Sanctuary model")
    temperature: float = Field(0.7, description="Sampling temperature (0.0 to 1.0)")
    max_tokens: int = Field(2048, description="Maximum number of tokens to generate")
    system_prompt: Optional[str] = Field(None, description="Optional system context to guide behavioral alignment")
