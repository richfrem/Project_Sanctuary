#!/usr/bin/env python3
"""
Forge LLM Models
=====================================

Purpose:
    Data definitions for Forge MCP operations.
    Defines response wrapper and request models.

Layer: Data (DTOs)

Key Models:
    # Internal / Responses
    - ModelQueryResponse: Standardized LLM response wrapper

    # MCP Requests
    - ForgeQueryRequest: Prompt and generation parameters
"""
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
