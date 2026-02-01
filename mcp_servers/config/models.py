#!/usr/bin/env python3
"""
Config Models
=====================================

Purpose:
    Data definitions for Config Server.
    Defines schemas for configuration items and MCP requests.

Layer: Data (DTOs)

Key Models:
    # Internal
    - ConfigItem: File metadata (name, size, modified)

    # MCP Requests
    - ConfigReadRequest
    - ConfigWriteRequest
    - ConfigDeleteRequest
"""

from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class ConfigItem:
    """Represents a configuration file metadata."""
    name: str
    size: int
    modified: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "size": self.size,
            "modified": self.modified
        }

#============================================
# FastMCP Request Models
#============================================
from pydantic import BaseModel, Field

class ConfigReadRequest(BaseModel):
    filename: str = Field(..., description="Name of the configuration file to read (e.g., 'mcp_servers.json')")

class ConfigWriteRequest(BaseModel):
    filename: str = Field(..., description="Name of the configuration file to write")
    content: str = Field(..., description="Content to write (string or JSON valid string)")

class ConfigDeleteRequest(BaseModel):
    filename: str = Field(..., description="Name of the configuration file to delete")
