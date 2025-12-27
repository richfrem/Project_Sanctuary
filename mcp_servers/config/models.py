#============================================
# mcp_servers/config/models.py
# Purpose: Data definition layer for Config Server.
# Role: Data Layer
# Used as: Type definitions for operations and validator.
# LIST OF CLASSES:
#   - ConfigItem (DataClass)
#============================================

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
