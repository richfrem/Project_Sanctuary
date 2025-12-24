#============================================
# mcp_servers/chronicle/validator.py
# Purpose: Validation logic for Chronicle entries.
#          Enforces data integrity, uniqueness, and modification windows.
# Role: Validation Layer
# Used as: Helper module by operations.py
# LIST OF CLASSES/FUNCTIONS:
#   - ChronicleValidator
#     - __init__
#     - get_next_entry_number
#     - validate_entry_number
#     - validate_modification_window
#     - validate_required_fields
#============================================
"""
Validation logic for Chronicle MCP.
"""
import os
import re
from datetime import datetime, date, timedelta
from typing import Optional
from .models import ChronicleStatus, ChronicleClassification

class ChronicleValidator:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir

    #============================================
    # Method: get_next_entry_number
    # Purpose: Determine the next sequential entry number.
    # Returns: Next available integer ID
    #============================================
    def get_next_entry_number(self) -> int:
        """Determine the next sequential entry number."""
        if not os.path.exists(self.base_dir):
            return 1
            
        files = os.listdir(self.base_dir)
        numbers = []
        for f in files:
            match = re.match(r"(\d{3})_", f)
            if match:
                numbers.append(int(match.group(1)))
        
        return max(numbers) + 1 if numbers else 1

    #============================================
    # Method: validate_entry_number
    # Purpose: Ensure entry number is unique for creation.
    # Args:
    #   number: Entry ID to check
    # Throws: ValueError if exists
    #============================================
    def validate_entry_number(self, number: int) -> None:
        """Ensure entry number is unique for creation."""
        if not os.path.exists(self.base_dir):
            return
            
        files = os.listdir(self.base_dir)
        for f in files:
            if f.startswith(f"{number:03d}_"):
                raise ValueError(f"Entry {number} already exists: {f}")

    #============================================
    # Method: validate_modification_window
    # Purpose: Enforce 7-day modification window.
    # Args:
    #   file_path: Path to entry file
    #   override_approval_id: Optional override ID
    #============================================
    def validate_modification_window(self, file_path: str, override_approval_id: Optional[str] = None) -> None:
        """
        Enforce 7-day modification window.
        Entries older than 7 days cannot be modified without override.
        """
        if not os.path.exists(file_path):
            return  # New file, always allowed
            
        # Check file creation/modification time or parse date from content
        # Using file modification time as a proxy for "age of entry" in filesystem
        # In a real system, we might parse the date from the file content
        
        stats = os.stat(file_path)
        last_mod = datetime.fromtimestamp(stats.st_mtime)
        age = datetime.now() - last_mod
        
        if age > timedelta(days=7):
            if not override_approval_id:
                raise ValueError(
                    f"Entry is {age.days} days old (limit: 7 days). "
                    "Modification requires 'override_approval_id'."
                )

    #============================================
    # Method: validate_required_fields
    # Purpose: Validate that required fields are present and not empty.
    # Args:
    #   title, content, author
    #============================================
    def validate_required_fields(self, title: str, content: str, author: str) -> None:
        """Validate that required fields are present and not empty."""
        if not title or not title.strip():
            raise ValueError("Title is required")
        if not content or not content.strip():
            raise ValueError("Content is required")
        if not author or not author.strip():
            raise ValueError("Author is required")
