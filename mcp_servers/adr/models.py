"""
ADR MCP Server - Data Models
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from datetime import datetime


class ADRStatus(Enum):
    """Valid ADR statuses."""
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    DEPRECATED = "deprecated"
    SUPERSEDED = "superseded"


@dataclass
class ADR:
    """ADR data model."""
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
