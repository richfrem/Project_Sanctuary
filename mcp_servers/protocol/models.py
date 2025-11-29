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
