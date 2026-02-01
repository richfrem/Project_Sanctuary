File operations for Protocol MCP.
"""
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from mcp_servers.lib.logging_utils import setup_mcp_logging

logger = setup_mcp_logging(__name__)
from .models import Protocol, ProtocolStatus, PROTOCOL_TEMPLATE
from .validator import ProtocolValidator


class ProtocolOperations:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.validator = ProtocolValidator(base_dir)
        
        # Ensure directory exists
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

    def create_protocol(
        self,
        number: Optional[int],
        title: str,
        status: str,
        classification: str,
        version: str,
        authority: str,
        content: str,
        linked_protocols: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a new protocol."""
        # Auto-generate number if not provided
        if number is None:
            number = self.validator.get_next_protocol_number()
        
        # Validate inputs
        self.validator.validate_required_fields(title, classification, version, authority, content)
        self.validator.validate_protocol_number(number)
        
        # Create protocol object
        protocol = Protocol(
            number=number,
            title=title,
            status=ProtocolStatus(status),
            classification=classification,
            version=version,
            authority=authority,
            content=content,
            linked_protocols=linked_protocols
        )
        
        # Generate linked protocols line
        linked_line = f"**Linked Protocols:** {linked_protocols}" if linked_protocols else ""
        
        # Generate content
        file_content = PROTOCOL_TEMPLATE.format(
            number=protocol.number,
            title=protocol.title,
            status=protocol.status.value,
            classification=protocol.classification,
            version=protocol.version,
            authority=protocol.authority,
            linked_protocols_line=linked_line,
            content=protocol.content
        )
        
        # Write file
        file_path = os.path.join(self.base_dir, protocol.filename)
        with open(file_path, "w") as f:
            f.write(file_content)
            
        return {
            "protocol_number": number,
            "file_path": file_path,
            "status": protocol.status.value
        }

    def update_protocol(
        self,
        number: int,
        updates: Dict[str, Any],
        reason: str
    ) -> Dict[str, Any]:
        """Update an existing protocol."""
        # Find file
        file_path = self._find_protocol_file(number)
        if not file_path:
            raise ValueError(f"Protocol {number} not found")
            
        # Read existing
        current_protocol = self.get_protocol(number)
        
        # Apply updates
        new_title = updates.get("title", current_protocol["title"])
        new_content = updates.get("content", current_protocol["content"])
        new_status = updates.get("status", current_protocol["status"])
        new_classification = updates.get("classification", current_protocol["classification"])
        new_version = updates.get("version", current_protocol["version"])
        new_authority = updates.get("authority", current_protocol["authority"])
        new_linked = updates.get("linked_protocols", current_protocol.get("linked_protocols", ""))
        
        # Generate linked protocols line
        linked_line = f"**Linked Protocols:** {new_linked}" if new_linked else ""
        
        # Re-generate content
        file_content = PROTOCOL_TEMPLATE.format(
            number=number,
            title=new_title,
            status=new_status,
            classification=new_classification,
            version=new_version,
            authority=new_authority,
            linked_protocols_line=linked_line,
            content=new_content
        )
        
        # Write file
        with open(file_path, "w") as f:
            f.write(file_content)
            
        return {
            "protocol_number": number,
            "updated_fields": list(updates.keys())
        }

    def get_protocol(self, number: int) -> Dict[str, Any]:
        """Retrieve a protocol."""
        file_path = self._find_protocol_file(number)
        if not file_path:
            raise ValueError(f"Protocol {number} not found")
            
        with open(file_path, "r") as f:
            content = f.read()
            
        return self._parse_protocol(content, number)

    def list_protocols(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List protocols."""
        if not os.path.exists(self.base_dir):
            return []
            
        files = sorted(os.listdir(self.base_dir))
        protocols = []
        
        for f in files:
            if not f.endswith(".md") or f.startswith("."):
                continue
                
            match = re.match(r"(\d+)_", f)
            if match:
                number = int(match.group(1))
                try:
                    protocol = self.get_protocol(number)
                    if status is None or protocol["status"] == status:
                        protocols.append(protocol)
                except Exception:
                    continue
                    
        return protocols

    def search_protocols(self, query: str) -> List[Dict[str, Any]]:
        """Search protocols."""
        if not os.path.exists(self.base_dir):
            return []
            
        results = []
        files = sorted(os.listdir(self.base_dir))
        
        for f in files:
            if not f.endswith(".md") or f.startswith("."):
                continue
                
            path = os.path.join(self.base_dir, f)
            with open(path, "r") as file:
                content = file.read()
                
            if query.lower() in content.lower():
                match = re.match(r"(\d+)_", f)
                if match:
                    number = int(match.group(1))
                    results.append(self._parse_protocol(content, number))
                    
        return results

    def _find_protocol_file(self, number: int) -> Optional[str]:
        """Find file path for a protocol number."""
        if not os.path.exists(self.base_dir):
            return None
            
        for f in os.listdir(self.base_dir):
            match = re.match(r"(\d+)_", f)
            if match and int(match.group(1)) == number:
                return os.path.join(self.base_dir, f)
        return None

    def _parse_protocol(self, content: str, number: int) -> Dict[str, Any]:
        """Parse markdown content into protocol dict."""
        lines = content.split("\n")
        metadata = {}
        body_start = 0
        
        for i, line in enumerate(lines):
            if line.startswith("**Status:**"):
                metadata["status"] = line.replace("**Status:**", "").strip()
            elif line.startswith("**Classification:**"):
                metadata["classification"] = line.replace("**Classification:**", "").strip()
            elif line.startswith("**Version:**"):
                metadata["version"] = line.replace("**Version:**", "").strip()
            elif line.startswith("**Authority:**"):
                metadata["authority"] = line.replace("**Authority:**", "").strip()
            elif line.startswith("**Linked Protocols:**"):
                metadata["linked_protocols"] = line.replace("**Linked Protocols:**", "").strip()
            elif line.strip() == "---":
                body_start = i + 1
                break
        
        # Extract title from H1
        title = "Unknown Protocol"
        for line in lines:
            if line.startswith("# Protocol"):
                parts = line.split(":", 1)
                if len(parts) > 1:
                    title = parts[1].strip()
                break
        
        return {
            "number": number,
            "title": title,
            "status": metadata.get("status", "PROPOSED"),
            "classification": metadata.get("classification", ""),
            "version": metadata.get("version", "1.0"),
            "authority": metadata.get("authority", ""),
            "linked_protocols": metadata.get("linked_protocols", ""),
            "content": "\n".join(lines[body_start:]).strip() if body_start > 0 else content
        }
