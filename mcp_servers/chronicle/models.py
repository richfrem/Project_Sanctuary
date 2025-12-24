#============================================
# mcp_servers/chronicle/models.py
# Purpose: Data models and constants for the Living Chronicle.
# Role: Data Definition Layer
# Used as: Type definition module by operations.py and server.py
# LIST OF CLASSES/CONSTANTS:
#   - ChronicleStatus (Enum)
#   - ChronicleClassification (Enum)
#   - ChronicleEntry (DataClass)
#   - CHRONICLE_TEMPLATE (Constant)
#============================================
"""
Data models for the Chronicle MCP server.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from datetime import date


class ChronicleStatus(str, Enum):
    DRAFT = "draft"
    PUBLISHED = "published"
    CANONICAL = "canonical"
    DEPRECATED = "deprecated"


class ChronicleClassification(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"


#============================================
#============================================
# DataClass: ChronicleEntry
# Purpose: Represents a single chronicle entry.
#============================================
@dataclass
class ChronicleEntry:
    entry_number: int
    title: str
    date: date
    author: str
    content: str
    status: ChronicleStatus = ChronicleStatus.DRAFT
    classification: ChronicleClassification = ChronicleClassification.INTERNAL
    
    #============================================
    # Property: filename
    # Purpose: Generate filename for the entry.
    # Returns: Formatted filename string
    #============================================
    @property
    def filename(self) -> str:
        """Generate filename for the entry."""
        # Format: 001_title_slug.md
        slug = self.title.lower().replace(" ", "_").replace("-", "_")
        # Remove non-alphanumeric chars except underscore
        slug = "".join(c for c in slug if c.isalnum() or c == "_")
        return f"{self.entry_number:03d}_{slug}.md"


CHRONICLE_TEMPLATE = """# Living Chronicle - Entry {number}

**Title:** {title}
**Date:** {date}
**Author:** {author}
**Status:** {status}
**Classification:** {classification}

---

{content}
"""

#============================================
# FastMCP Request Models
#============================================
from pydantic import BaseModel, Field
from typing import Dict, Any

class ChronicleCreateRequest(BaseModel):
    title: str = Field(..., description="Entry title")
    content: str = Field(..., description="Entry content (markdown)")
    author: str = Field(..., description="Author name/ID")
    date: Optional[str] = Field(None, description="Date string (YYYY-MM-DD), defaults to today")
    status: str = Field("draft", description="draft, published, canonical, deprecated")
    classification: str = Field("internal", description="public, internal, confidential")

class ChronicleUpdateRequest(BaseModel):
    entry_number: int = Field(..., description="The entry number to update")
    updates: Dict[str, Any] = Field(..., description="Dictionary of fields to update")
    reason: str = Field(..., description="Reason for the update")
    override_approval_id: Optional[str] = Field(None, description="Required if entry is older than 7 days")

class ChronicleGetRequest(BaseModel):
    entry_number: int = Field(..., description="The entry number to retrieve")

class ChronicleListRequest(BaseModel):
    limit: int = Field(10, description="Maximum number of entries to return")

class ChronicleSearchRequest(BaseModel):
    query: str = Field(..., description="Search query string")
