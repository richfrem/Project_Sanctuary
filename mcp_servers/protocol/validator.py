#============================================
# mcp_servers/protocol/validator.py
# Purpose: Validation logic for Protocol MCP.
#          Ensures protocol integrity and uniqueness.
# Role: Validation Layer
# Used as: Helper for Operations.
#============================================

"""
Validation logic for Protocol MCP.
"""
import os
import re
from typing import Optional
from .models import ProtocolStatus


class ProtocolValidator:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    def validate_protocol_number(self, number: int) -> None:
        """Ensure protocol number is unique."""
        if not os.path.exists(self.base_dir):
            return
            
        files = os.listdir(self.base_dir)
        for f in files:
            # Match files like "00_Title.md" or "100_Title.md"
            match = re.match(r"(\d+)_", f)
            if match and int(match.group(1)) == number:
                raise ValueError(f"Protocol {number} already exists: {f}")

    def validate_required_fields(
        self, 
        title: str, 
        classification: str, 
        version: str, 
        authority: str, 
        content: str
    ) -> None:
        """Validate that required fields are present and not empty."""
        if not title or not title.strip():
            raise ValueError("Title is required")
        if not classification or not classification.strip():
            raise ValueError("Classification is required")
        if not version or not version.strip():
            raise ValueError("Version is required")
        if not authority or not authority.strip():
            raise ValueError("Authority is required")
        if not content or not content.strip():
            raise ValueError("Content is required")
