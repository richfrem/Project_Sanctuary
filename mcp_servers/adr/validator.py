"""
ADR MCP Server - Validation Logic
"""
import os
import re
from typing import List, Optional
from .models import ADRStatus, VALID_TRANSITIONS


class ADRValidator:
    """Validates ADR operations."""
    
    def __init__(self, adrs_dir: str = "ADRs"):
        self.adrs_dir = adrs_dir
    
    def get_next_adr_number(self) -> int:
        """Get the next sequential ADR number."""
        if not os.path.exists(self.adrs_dir):
            return 1
        
        existing_numbers = []
        for filename in os.listdir(self.adrs_dir):
            if filename.endswith('.md') and not filename.startswith('adr_schema'):
                match = re.match(r'^(\d{3})_', filename)
                if match:
                    existing_numbers.append(int(match.group(1)))
        
        if not existing_numbers:
            return 1
        
        return max(existing_numbers) + 1
    
    def validate_adr_number(self, number: int) -> None:
        """Validate ADR number doesn't already exist."""
        filename_pattern = f"{number:03d}_*.md"
        for filename in os.listdir(self.adrs_dir):
            if re.match(f"^{number:03d}_", filename):
                raise ValueError(f"ADR {number:03d} already exists: {filename}")
    
    def validate_status_transition(
        self, 
        current_status: ADRStatus, 
        new_status: ADRStatus
    ) -> None:
        """Validate status transition is allowed."""
        if current_status == new_status:
            return  # No change is always valid
        
        allowed = VALID_TRANSITIONS.get(current_status, [])
        if new_status not in allowed:
            raise ValueError(
                f"Invalid transition from '{current_status.value}' to '{new_status.value}'. "
                f"Allowed transitions: {[s.value for s in allowed]}"
            )
    
    def validate_supersedes(self, supersedes: Optional[int]) -> None:
        """Validate that superseded ADR exists."""
        if supersedes is None:
            return
        
        # Check if the ADR exists
        found = False
        for filename in os.listdir(self.adrs_dir):
            if re.match(f"^{supersedes:03d}_", filename):
                found = True
                break
        
        if not found:
            raise ValueError(
                f"ADR {supersedes:03d} does not exist (referenced in supersedes)"
            )
    
    def validate_required_fields(
        self,
        title: str,
        context: str,
        decision: str,
        consequences: str
    ) -> None:
        """Validate required fields are not empty."""
        if not title or not title.strip():
            raise ValueError("Title is required")
        if not context or not context.strip():
            raise ValueError("Context is required")
        if not decision or not decision.strip():
            raise ValueError("Decision is required")
        if not consequences or not consequences.strip():
            raise ValueError("Consequences are required")
