"""
File operations for Chronicle MCP.
"""
import os
import re
from datetime import date, datetime
from typing import List, Optional, Dict, Any
from pathlib import Path
import sys

# Setup logging
sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.logging_utils import setup_mcp_logging

logger = setup_mcp_logging(__name__)

from .models import ChronicleEntry, ChronicleStatus, ChronicleClassification, CHRONICLE_TEMPLATE
from .validator import ChronicleValidator


class ChronicleOperations:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        self.validator = ChronicleValidator(base_dir)
        
        # Ensure directory exists
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

    def create_entry(
        self,
        title: str,
        content: str,
        author: str,
        date_str: Optional[str] = None,
        status: str = "draft",
        classification: str = "internal"
    ) -> Dict[str, Any]:
        """Create a new chronicle entry."""
        # Validate inputs
        self.validator.validate_required_fields(title, content, author)
        
        # Determine number
        number = self.validator.get_next_entry_number()
        self.validator.validate_entry_number(number)
        
        # Parse date
        entry_date = date.fromisoformat(date_str) if date_str else date.today()
        
        # Create entry object
        entry = ChronicleEntry(
            entry_number=number,
            title=title,
            date=entry_date,
            author=author,
            content=content,
            status=ChronicleStatus(status),
            classification=ChronicleClassification(classification)
        )
        
        # Generate content
        file_content = CHRONICLE_TEMPLATE.format(
            number=entry.entry_number,
            title=entry.title,
            date=entry.date.isoformat(),
            author=entry.author,
            status=entry.status.value,
            classification=entry.classification.value,
            content=entry.content
        )
        
        # Write file
        file_path = os.path.join(self.base_dir, entry.filename)
        with open(file_path, "w") as f:
            f.write(file_content)
            
        return {
            "entry_number": number,
            "file_path": file_path,
            "status": entry.status.value
        }

    def update_entry(
        self,
        entry_number: int,
        updates: Dict[str, Any],
        reason: str,
        override_approval_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update an existing chronicle entry."""
        # Find file
        file_path = self._find_entry_file(entry_number)
        if not file_path:
            raise ValueError(f"Entry {entry_number} not found")
            
        # Validate modification window
        self.validator.validate_modification_window(file_path, override_approval_id)
        
        # Read existing
        current_entry = self.get_entry(entry_number)
        
        # Apply updates
        # Note: In a real implementation, we'd need to parse the markdown back into an object
        # For now, we'll re-generate the file with updated fields
        
        new_title = updates.get("title", current_entry["title"])
        new_content = updates.get("content", current_entry["content"])
        new_status = updates.get("status", current_entry["status"])
        new_classification = updates.get("classification", current_entry["classification"])
        
        # Re-generate content
        file_content = CHRONICLE_TEMPLATE.format(
            number=entry_number,
            title=new_title,
            date=current_entry["date"],
            author=current_entry["author"], # Author usually doesn't change
            status=new_status,
            classification=new_classification,
            content=new_content
        )
        
        # Write file
        with open(file_path, "w") as f:
            f.write(file_content)
            
        # If title changed, we might need to rename the file, but let's keep it simple for now
        # and only update content. Renaming would break links.
        
        return {
            "entry_number": entry_number,
            "updated_fields": list(updates.keys())
        }

    def get_entry(self, entry_number: int) -> Dict[str, Any]:
        """Retrieve a chronicle entry."""
        file_path = self._find_entry_file(entry_number)
        if not file_path:
            raise ValueError(f"Entry {entry_number} not found")
            
        with open(file_path, "r") as f:
            content = f.read()
            
        return self._parse_entry(content, entry_number)

    def list_entries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent chronicle entries."""
        if not os.path.exists(self.base_dir):
            return []
            
        files = sorted(os.listdir(self.base_dir), reverse=True)
        entries = []
        
        for f in files:
            if not f.endswith(".md"):
                continue
                
            match = re.match(r"(\d{3})_", f)
            if match:
                number = int(match.group(1))
                try:
                    entries.append(self.get_entry(number))
                except Exception:
                    continue # Skip malformed
                    
            if len(entries) >= limit:
                break
                
        return entries

    def search_entries(self, query: str) -> List[Dict[str, Any]]:
        """Search chronicle entries."""
        if not os.path.exists(self.base_dir):
            return []
            
        results = []
        files = sorted(os.listdir(self.base_dir))
        
        for f in files:
            if not f.endswith(".md"):
                continue
                
            path = os.path.join(self.base_dir, f)
            with open(path, "r") as file:
                content = file.read()
                
            if query.lower() in content.lower():
                match = re.match(r"(\d{3})_", f)
                if match:
                    number = int(match.group(1))
                    results.append(self._parse_entry(content, number))
                    
        return results

    def _find_entry_file(self, number: int) -> Optional[str]:
        """Find file path for an entry number."""
        if not os.path.exists(self.base_dir):
            return None
            
        for f in os.listdir(self.base_dir):
            if f.startswith(f"{number:03d}_"):
                return os.path.join(self.base_dir, f)
        return None

    def _parse_entry(self, content: str, number: int) -> Dict[str, Any]:
        """Parse markdown content into entry dict."""
        # Simple parsing logic
        # Extract metadata from lines
        lines = content.split("\n")
        metadata = {}
        body_start = 0
        
        for i, line in enumerate(lines):
            if line.startswith("**Title:**"):
                metadata["title"] = line.replace("**Title:**", "").strip()
            elif line.startswith("**Date:**"):
                metadata["date"] = line.replace("**Date:**", "").strip()
            elif line.startswith("**Author:**"):
                metadata["author"] = line.replace("**Author:**", "").strip()
            elif line.startswith("**Status:**"):
                metadata["status"] = line.replace("**Status:**", "").strip()
            elif line.startswith("**Classification:**"):
                metadata["classification"] = line.replace("**Classification:**", "").strip()
            elif line.strip() == "---":
                body_start = i + 1
                break
        
        # Fallback for older formats if title not found in metadata
        if "title" not in metadata:
             # Try to find H1 or H3
             for line in lines:
                 if line.startswith("# "):
                     metadata["title"] = line.replace("# ", "").replace("Living Chronicle - Entry " + str(number), "").strip()
                     break
                 elif line.startswith("### **Entry"):
                     # Format: ### **Entry 001: The Genesis...**
                     parts = line.split(":")
                     if len(parts) > 1:
                         metadata["title"] = parts[1].replace("**", "").strip()
                     break

        return {
            "number": number,
            "title": metadata.get("title", "Unknown Title"),
            "date": metadata.get("date", ""),
            "author": metadata.get("author", ""),
            "status": metadata.get("status", "draft"),
            "classification": metadata.get("classification", "internal"),
            "content": "\n".join(lines[body_start:]).strip() if body_start > 0 else content
        }
