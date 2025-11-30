from fastmcp import FastMCP
import os
from typing import Optional, Dict, Any, List
from .operations import ProtocolOperations

# Initialize FastMCP
mcp = FastMCP("project_sanctuary.protocol")

# Configuration
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
PROTOCOL_DIR = os.path.join(PROJECT_ROOT, "01_PROTOCOLS")

# Initialize operations
ops = ProtocolOperations(PROTOCOL_DIR)


@mcp.tool()
def protocol_create(
    number: int,
    title: str,
    status: str,
    classification: str,
    version: str,
    authority: str,
    content: str,
    linked_protocols: Optional[str] = None
) -> str:
    """
    Create a new protocol.
    
    Args:
        number: Protocol number (e.g., 117)
        title: Protocol title
        status: PROPOSED, CANONICAL, or DEPRECATED
        classification: Classification (e.g., "Foundational Framework")
        version: Version string (e.g., "1.0")
        authority: Authority/author
        content: Protocol content (markdown)
        linked_protocols: Optional linked protocol references
    """
    try:
        result = ops.create_protocol(
            number, title, status, classification, version, authority, content, linked_protocols
        )
        return f"Created Protocol {result['protocol_number']}: {result['file_path']}"
    except Exception as e:
        return f"Error creating protocol: {str(e)}"


@mcp.tool()
def protocol_update(
    number: int,
    updates: Dict[str, Any],
    reason: str
) -> str:
    """
    Update an existing protocol.
    
    Args:
        number: Protocol number to update
        updates: Dictionary of fields to update
        reason: Reason for the update
    """
    try:
        result = ops.update_protocol(number, updates, reason)
        return f"Updated Protocol {result['protocol_number']}. Fields: {', '.join(result['updated_fields'])}"
    except Exception as e:
        return f"Error updating protocol: {str(e)}"


@mcp.tool()
def protocol_get(number: int) -> str:
    """
    Retrieve a specific protocol.
    
    Args:
        number: Protocol number to retrieve
    """
    try:
        protocol = ops.get_protocol(number)
        return f"""Protocol {protocol['number']}: {protocol['title']}
Status: {protocol['status']}
Classification: {protocol['classification']}
Version: {protocol['version']}
Authority: {protocol['authority']}
Linked Protocols: {protocol.get('linked_protocols', 'None')}

{protocol['content']}"""
    except Exception as e:
        return f"Error retrieving protocol: {str(e)}"


@mcp.tool()
def protocol_list(status: Optional[str] = None) -> str:
    """
    List protocols.
    
    Args:
        status: Optional status filter (PROPOSED, CANONICAL, DEPRECATED)
    """
    try:
        protocols = ops.list_protocols(status)
        if not protocols:
            return "No protocols found."
            
        output = [f"Found {len(protocols)} protocol(s):"]
        for p in protocols:
            output.append(f"- {p['number']:03d}: {p['title']} [{p['status']}] v{p['version']}")
        return "\n".join(output)
    except Exception as e:
        return f"Error listing protocols: {str(e)}"


@mcp.tool()
def protocol_search(query: str) -> str:
    """
    Search protocols by content.
    
    Args:
        query: Search query string
    """
    try:
        results = ops.search_protocols(query)
        if not results:
            return f"No protocols found matching '{query}'"
            
        output = [f"Found {len(results)} protocol(s) matching '{query}':"]
        for r in results:
            output.append(f"- {r['number']:03d}: {r['title']}")
        return "\n".join(output)
    except Exception as e:
        return f"Error searching protocols: {str(e)}"


if __name__ == "__main__":
    mcp.run()
