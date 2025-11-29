"""
ADR MCP Server - File Operations
"""
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from .models import ADR, ADRStatus, ADR_TEMPLATE
from .validator import ADRValidator


class ADROperations:
    """Handles ADR file operations."""
    
    def __init__(self, adrs_dir: str = "ADRs"):
        self.adrs_dir = adrs_dir
        self.validator = ADRValidator(adrs_dir)
        
        # Ensure directory exists
        os.makedirs(self.adrs_dir, exist_ok=True)
    
    def create_adr(
        self,
        title: str,
        context: str,
        decision: str,
        consequences: str,
        date: Optional[str] = None,
        status: str = "proposed",
        author: str = "AI Assistant",
        supersedes: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create a new ADR."""
        # Validate inputs
        self.validator.validate_required_fields(title, context, decision, consequences)
        self.validator.validate_supersedes(supersedes)
        
        # Get next number
        adr_number = self.validator.get_next_adr_number()
        
        # Use current date if not provided
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        # Create filename from title
        filename_title = title.lower().replace(" ", "_")
        filename_title = re.sub(r'[^a-z0-9_]', '', filename_title)
        filename = f"{adr_number:03d}_{filename_title}.md"
        filepath = os.path.join(self.adrs_dir, filename)
        
        # Format context line
        context_line = ""
        if supersedes:
            context_line = f"**Supersedes:** ADR {supersedes:03d}"
        
        # Generate content from template
        content = ADR_TEMPLATE.format(
            title=title,
            status=status,
            date=date,
            author=author,
            context_line=context_line,
            context=context,
            decision=decision,
            consequences=consequences
        )
        
        # Write file
        with open(filepath, 'w') as f:
            f.write(content)
        
        return {
            "adr_number": adr_number,
            "file_path": filepath,
            "status": status
        }
    
    def update_adr_status(
        self,
        number: int,
        new_status: str,
        reason: str
    ) -> Dict[str, Any]:
        """Update ADR status."""
        # Find the ADR file
        filepath = self._find_adr_file(number)
        if not filepath:
            raise FileNotFoundError(f"ADR {number:03d} not found")
        
        # Read current content
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Extract current status
        status_match = re.search(r'\*\*Status:\*\* (\w+)', content)
        if not status_match:
            raise ValueError(f"Could not find status in ADR {number:03d}")
        
        old_status = ADRStatus(status_match.group(1))
        new_status_enum = ADRStatus(new_status)
        
        # Validate transition
        self.validator.validate_status_transition(old_status, new_status_enum)
        
        # Update status in content
        updated_content = re.sub(
            r'\*\*Status:\*\* \w+',
            f'**Status:** {new_status}',
            content
        )
        
        # Add update note
        update_note = f"\n\n---\n\n**Status Update ({datetime.now().strftime('%Y-%m-%d')}):** {reason}\n"
        updated_content += update_note
        
        # Write back
        with open(filepath, 'w') as f:
            f.write(updated_content)
        
        return {
            "adr_number": number,
            "old_status": old_status.value,
            "new_status": new_status,
            "updated_at": datetime.now().strftime("%Y-%m-%d")
        }
    
    def get_adr(self, number: int) -> Dict[str, Any]:
        """Get a specific ADR."""
        filepath = self._find_adr_file(number)
        if not filepath:
            raise FileNotFoundError(f"ADR {number:03d} not found")
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Parse ADR content
        title_match = re.search(r'^# (.+)$', content, re.MULTILINE)
        status_match = re.search(r'\*\*Status:\*\* (\w+)', content)
        date_match = re.search(r'\*\*Date:\*\* ([\d-]+)', content)
        author_match = re.search(r'\*\*Author:\*\* (.+)$', content, re.MULTILINE)
        
        context_match = re.search(r'## Context\n\n(.+?)(?=\n## )', content, re.DOTALL)
        decision_match = re.search(r'## Decision\n\n(.+?)(?=\n## )', content, re.DOTALL)
        consequences_match = re.search(r'## Consequences\n\n(.+?)(?=\n---|$)', content, re.DOTALL)
        
        return {
            "number": number,
            "title": title_match.group(1) if title_match else "Unknown",
            "status": status_match.group(1) if status_match else "unknown",
            "date": date_match.group(1) if date_match else "unknown",
            "author": author_match.group(1) if author_match else "Unknown",
            "context": context_match.group(1).strip() if context_match else "",
            "decision": decision_match.group(1).strip() if decision_match else "",
            "consequences": consequences_match.group(1).strip() if consequences_match else "",
            "file_path": filepath
        }
    
    def list_adrs(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all ADRs."""
        adrs = []
        
        for filename in sorted(os.listdir(self.adrs_dir)):
            if not filename.endswith('.md') or filename.startswith('adr_schema'):
                continue
            
            match = re.match(r'^(\d{3})_', filename)
            if not match:
                continue
            
            number = int(match.group(1))
            try:
                adr = self.get_adr(number)
                
                # Filter by status if provided
                if status and adr['status'] != status:
                    continue
                
                adrs.append({
                    "number": adr['number'],
                    "title": adr['title'],
                    "status": adr['status'],
                    "date": adr['date']
                })
            except Exception:
                continue
        
        return adrs
    
    def search_adrs(self, query: str) -> List[Dict[str, Any]]:
        """Search ADRs by content."""
        results = []
        query_lower = query.lower()
        
        for filename in sorted(os.listdir(self.adrs_dir)):
            if not filename.endswith('.md') or filename.startswith('adr_schema'):
                continue
            
            match = re.match(r'^(\d{3})_', filename)
            if not match:
                continue
            
            number = int(match.group(1))
            filepath = os.path.join(self.adrs_dir, filename)
            
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Search in content
            if query_lower in content.lower():
                # Extract matching lines
                matches = []
                for line in content.split('\n'):
                    if query_lower in line.lower():
                        matches.append(line.strip())
                        if len(matches) >= 3:  # Limit to 3 matches per ADR
                            break
                
                adr = self.get_adr(number)
                results.append({
                    "number": number,
                    "title": adr['title'],
                    "matches": matches
                })
        
        return results
    
    def _find_adr_file(self, number: int) -> Optional[str]:
        """Find ADR file by number."""
        for filename in os.listdir(self.adrs_dir):
            if re.match(f"^{number:03d}_", filename):
                return os.path.join(self.adrs_dir, filename)
        return None
