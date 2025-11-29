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


@dataclass
class ChronicleEntry:
    entry_number: int
    title: str
    date: date
    author: str
    content: str
    status: ChronicleStatus = ChronicleStatus.DRAFT
    classification: ChronicleClassification = ChronicleClassification.INTERNAL
    
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
