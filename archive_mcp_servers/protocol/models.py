#!/usr/bin/env python3
"""
Protocol Models
=====================================

Purpose:
    Data models for the Protocol MCP server.
    Defines Protocol dataclass and Status enum.

Layer: Data (DTOs)

Key Models:
    # Internal / Enums
    - ProtocolStatus (Enum): PROPOSED, CANONICAL, DEPRECATED
    - Protocol: Internal representation
        - filename (property)

    # MCP Requests
    - ProtocolCreateRequest
    - ProtocolUpdateRequest
    - ProtocolGetRequest
    - ProtocolListRequest
    - ProtocolSearchRequest
"""

"""
Data models for the Protocol MCP server.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ProtocolStatus(str, Enum):
    PROPOSED = "PROPOSED"
    CANONICAL = "CANONICAL"
    DEPRECATED = "DEPRECATED"


@dataclass
class Protocol:
    number: int
    title: str
    status: ProtocolStatus
    classification: str
    version: str
    authority: str
    content: str
    linked_protocols: Optional[str] = None
    
    @property
    def filename(self) -> str:
        """Generate filename for the protocol."""
        # Format: 00_Protocol_Title.md
        slug = self.title.replace(" ", "_").replace("-", "_")
        # Remove non-alphanumeric chars except underscore
        slug = "".join(c for c in slug if c.isalnum() or c == "_")
        return f"{self.number:02d}_{slug}.md"


PROTOCOL_TEMPLATE = """# Protocol {number}: {title}

**Status:** {status}
**Classification:** {classification}
**Version:** {version}
**Authority:** {authority}
{linked_protocols_line}
---

{content}
"""

#============================================
# FastMCP Request Models
#============================================
from pydantic import BaseModel, Field
from typing import Dict, Any

class ProtocolCreateRequest(BaseModel):
    number: int = Field(..., description="Unique protocol number")
    title: str = Field(..., description="Descriptive title of the protocol")
    status: str = Field(..., description="Status (PROPOSED, CANONICAL, DEPRECATED)")
    classification: str = Field(..., description="Classification (Internal, Public, etc.)")
    version: str = Field(..., description="Version string (e.g., 1.0)")
    authority: str = Field(..., description="Authorizing entity or role")
    content: str = Field(..., description="Full protocol content in markdown")
    linked_protocols: Optional[str] = Field(None, description="Comma-separated list of related protocol numbers")

class ProtocolUpdateRequest(BaseModel):
    number: int = Field(..., description="Protocol number to update")
    updates: Dict[str, Any] = Field(..., description="Dictionary of fields to change")
    reason: str = Field(..., description="Justification for the update")

class ProtocolGetRequest(BaseModel):
    number: int = Field(..., description="Protocol number to retrieve")

class ProtocolListRequest(BaseModel):
    status: Optional[str] = Field(None, description="Filter protocols by status")

class ProtocolSearchRequest(BaseModel):
    query: str = Field(..., description="Search term or regex pattern")
