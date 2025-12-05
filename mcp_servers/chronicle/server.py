from fastmcp import FastMCP
import os
from typing import Optional, List, Dict, Any
from mcp_servers.chronicle.operations import ChronicleOperations

# Initialize FastMCP
mcp = FastMCP("project_sanctuary.chronicle")

# Configuration
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
CHRONICLE_DIR = os.path.join(PROJECT_ROOT, "00_CHRONICLE/ENTRIES")

# Initialize operations
ops = ChronicleOperations(CHRONICLE_DIR)


@mcp.tool()
def chronicle_create_entry(
    title: str,
    content: str,
    author: str,
    date: Optional[str] = None,
    status: str = "draft",
    classification: str = "internal"
) -> str:
    """
    Create a new chronicle entry.
    
    Args:
        title: Entry title
        content: Entry content (markdown)
        author: Author name/ID
        date: Date string (YYYY-MM-DD), defaults to today
        status: draft, published, canonical, deprecated
        classification: public, internal, confidential
    """
    try:
        result = ops.create_entry(title, content, author, date, status, classification)
        return f"Created Chronicle Entry {result['entry_number']}: {result['file_path']}"
    except Exception as e:
        return f"Error creating entry: {str(e)}"


@mcp.tool()
def chronicle_append_entry(
    title: str,
    content: str,
    author: str,
    date: Optional[str] = None,
    status: str = "draft",
    classification: str = "internal"
) -> str:
    """
    Append a new entry to the Chronicle (Alias for create_entry).
    
    Args:
        title: Entry title
        content: Entry content
        author: Author name
        date: Date string
        status: Status
        classification: Classification
    """
    try:
        result = ops.create_entry(title, content, author, date, status, classification)
        return f"Created Chronicle Entry {result['entry_number']}: {result['file_path']}"
    except Exception as e:
        return f"Error creating entry: {str(e)}"



@mcp.tool()
def chronicle_update_entry(
    entry_number: int,
    updates: Dict[str, Any],
    reason: str,
    override_approval_id: Optional[str] = None
) -> str:
    """
    Update an existing chronicle entry.
    
    Args:
        entry_number: The entry number to update
        updates: Dictionary of fields to update (title, content, status, classification)
        reason: Reason for the update
        override_approval_id: Required if entry is older than 7 days
    """
    try:
        result = ops.update_entry(entry_number, updates, reason, override_approval_id)
        return f"Updated Chronicle Entry {result['entry_number']}. Fields: {', '.join(result['updated_fields'])}"
    except Exception as e:
        return f"Error updating entry: {str(e)}"


@mcp.tool()
def chronicle_get_entry(entry_number: int) -> str:
    """
    Retrieve a specific chronicle entry.
    
    Args:
        entry_number: The entry number to retrieve
    """
    try:
        entry = ops.get_entry(entry_number)
        return f"""Entry {entry['number']}: {entry['title']}
Date: {entry['date']}
Author: {entry['author']}
Status: {entry['status']}
Classification: {entry['classification']}

{entry['content']}"""
    except Exception as e:
        return f"Error retrieving entry: {str(e)}"


@mcp.tool()
def chronicle_list_entries(limit: int = 10) -> str:
    """
    List recent chronicle entries.
    
    Args:
        limit: Maximum number of entries to return (default 10)
    """
    try:
        entries = ops.list_entries(limit)
        if not entries:
            return "No entries found."
            
        output = [f"Found {len(entries)} recent entries:"]
        for e in entries:
            output.append(f"- {e['number']:03d}: {e['title']} [{e['status']}] ({e['date']})")
        return "\n".join(output)
    except Exception as e:
        return f"Error listing entries: {str(e)}"


@mcp.tool()
def chronicle_read_latest_entries(limit: int = 10) -> str:
    """
    Read the latest entries from the Chronicle (Alias for list_entries).
    
    Args:
        limit: Number of entries to read
    """
    try:
        entries = ops.list_entries(limit)
        if not entries:
            return "No entries found."
            
        output = [f"Found {len(entries)} recent entries:"]
        for e in entries:
            output.append(f"- {e['number']:03d}: {e['title']} [{e['status']}] ({e['date']})")
        return "\n".join(output)
    except Exception as e:
        return f"Error listing entries: {str(e)}"



@mcp.tool()
def chronicle_search(query: str) -> str:
    """
    Search chronicle entries by content.
    
    Args:
        query: Search query string
    """
    try:
        results = ops.search_entries(query)
        if not results:
            return f"No entries found matching '{query}'"
            
        output = [f"Found {len(results)} entries matching '{query}':"]
        for r in results:
            output.append(f"- {r['number']:03d}: {r['title']}")
        return "\n".join(output)
    except Exception as e:
        return f"Error searching entries: {str(e)}"


if __name__ == "__main__":
    mcp.run()
