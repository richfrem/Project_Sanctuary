"""
Forge MCP Models
Domain: project_sanctuary.system.forge

Data models for Forge MCP operations.
"""
from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class ModelQueryResponse:
    """Response from querying the Sanctuary model."""
    model: str
    response: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    temperature: Optional[float] = None
    status: str = "success"
    error: Optional[str] = None


def to_dict(obj: Any) -> Dict[str, Any]:
    """Convert dataclass to dictionary."""
    if hasattr(obj, '__dataclass_fields__'):
        return {k: v for k, v in obj.__dict__.items() if v is not None}
    return obj
