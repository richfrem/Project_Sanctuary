#============================================
# mcp_servers/adr/models.py
# Purpose: Data models for the ADR System.
# Role: Data Definition Layer
# Used as: Type definition module by operations.py and server.py
# LIST OF CLASSES/CONSTANTS:
#   - ADRStatus (Enum)
#   - ADR (DataClass)
#   - ADR_TEMPLATE (Constant)
#   - VALID_TRANSITIONS (Constant)
#============================================
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from datetime import datetime


#============================================
# Enum: ADRStatus
# Purpose: Define valid workflow states.
#============================================
class ADRStatus(Enum):
    """ADR Status enumeration."""
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    DEPRECATED = "deprecated"
    SUPERSEDED = "superseded"


#============================================
# DataClass: ADR
# Purpose: Structured representation of an Architecture Decision Record.
#============================================
@dataclass
class ADR:
    """ADR data class."""
    number: int
    title: str
    status: ADRStatus
    date: str
    author: str
    context: str
    decision: str
    consequences: str
    supersedes: Optional[int] = None
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            "number": self.number,
            "title": self.title,
            "status": self.status.value,
            "date": self.date,
            "author": self.author,
            "context": self.context,
            "decision": self.decision,
            "consequences": self.consequences,
            "supersedes": self.supersedes
        }


# ADR Template
ADR_TEMPLATE = """# {title}

**Status:** {status}
**Date:** {date}
**Author:** {author}
{context_line}

---

## Context

{context}

## Decision

{decision}

## Consequences

{consequences}
"""


# Valid status transitions
VALID_TRANSITIONS = {
    ADRStatus.PROPOSED: [ADRStatus.ACCEPTED, ADRStatus.DEPRECATED],
    ADRStatus.ACCEPTED: [ADRStatus.DEPRECATED, ADRStatus.SUPERSEDED],
    ADRStatus.DEPRECATED: [],
    ADRStatus.SUPERSEDED: []
}

#============================================
# FastMCP Request Models
#============================================
from pydantic import BaseModel, Field

class ADRCreateRequest(BaseModel):
    title: str = Field(..., description="Short, descriptive title of the architectural decision")
    context: str = Field(..., description="The problem description, background, and force factors")
    decision: str = Field(..., description="The specific decision made and why")
    consequences: str = Field(..., description="The positive and negative outcomes of the decision")
    date: Optional[str] = Field(None, description="ISO date (YYYY-MM-DD), defaults to today")
    status: str = Field("proposed", description="Initial status: proposed, accepted, deprecated, superseded")
    author: str = Field("AI Assistant", description="Person/Agent making the decision")
    supersedes: Optional[int] = Field(None, description="The ADR number this new decision replaces")

class ADRUpdateStatusRequest(BaseModel):
    number: int = Field(..., description="The numeric ID of the ADR (e.g., 66)")
    new_status: str = Field(..., description="New status value")
    reason: str = Field(..., description="Brief explanation for the status change")

class ADRGetRequest(BaseModel):
    number: int = Field(..., description="The numeric ID of the ADR to retrieve")

class ADRListRequest(BaseModel):
    status: Optional[str] = Field(None, description="Optional filter by status (e.g., 'accepted')")

class ADRSearchRequest(BaseModel):
    query: str = Field(..., description="Search keyword or phrase across ADR titles and content")
