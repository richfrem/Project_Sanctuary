"""
ADR MCP Server
Domain: project_sanctuary.document.adr
"""
from fastmcp import FastMCP
from .operations import ADROperations
import os
from typing import Optional

# Initialize FastMCP with canonical domain name
mcp = FastMCP("project_sanctuary.document.adr")

# Initialize ADR operations
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", ".")
ADRS_DIR = os.path.join(PROJECT_ROOT, "ADRs")
adr_ops = ADROperations(ADRS_DIR)


@mcp.tool()
def adr_create(
    title: str,
    context: str,
    decision: str,
    consequences: str,
    date: Optional[str] = None,
    status: str = "proposed",
    author: str = "AI Assistant",
    supersedes: Optional[int] = None
) -> str:
    """
    Create a new ADR with automatic sequential numbering.
    
    Args:
        title: ADR title
        context: Problem description and background
        decision: What was decided and why
        consequences: Positive/negative outcomes and risks
        date: Decision date (defaults to today)
        status: Initial status (defaults to "proposed")
        author: Decision maker (defaults to "AI Assistant")
        supersedes: ADR number this supersedes (optional)
        
    Returns:
        JSON string with adr_number, file_path, and status
        
    Example:
        adr_create(
            title="Adopt FastAPI for REST APIs",
            context="Need a modern Python web framework for building REST APIs...",
            decision="We will use FastAPI for all new REST API development...",
            consequences="Positive: Fast, modern, async support. Negative: Learning curve."
        )
    """
    try:
        result = adr_ops.create_adr(
            title=title,
            context=context,
            decision=decision,
            consequences=consequences,
            date=date,
            status=status,
            author=author,
            supersedes=supersedes
        )
        return f"Created ADR {result['adr_number']:03d}: {result['file_path']}"
    except Exception as e:
        return f"Error creating ADR: {str(e)}"


@mcp.tool()
def adr_update_status(number: int, new_status: str, reason: str) -> str:
    """
    Update the status of an existing ADR.
    
    Valid transitions:
    - proposed → accepted
    - proposed → deprecated
    - accepted → deprecated
    - accepted → superseded
    
    Args:
        number: ADR number
        new_status: New status (proposed/accepted/deprecated/superseded)
        reason: Reason for status change
        
    Returns:
        Status update confirmation
        
    Example:
        adr_update_status(
            number=38,
            new_status="accepted",
            reason="Implemented and tested successfully"
        )
    """
    try:
        result = adr_ops.update_adr_status(number, new_status, reason)
        return (
            f"Updated ADR {result['adr_number']:03d}: "
            f"{result['old_status']} → {result['new_status']} "
            f"(Reason: {reason})"
        )
    except Exception as e:
        return f"Error updating ADR status: {str(e)}"


@mcp.tool()
def adr_get(number: int) -> str:
    """
    Retrieve a specific ADR by number.
    
    Args:
        number: ADR number
        
    Returns:
        ADR details including title, status, context, decision, and consequences
        
    Example:
        adr_get(37)
    """
    try:
        adr = adr_ops.get_adr(number)
        return (
            f"ADR {adr['number']:03d}: {adr['title']}\n"
            f"Status: {adr['status']}\n"
            f"Date: {adr['date']}\n"
            f"Author: {adr['author']}\n\n"
            f"Context:\n{adr['context']}\n\n"
            f"Decision:\n{adr['decision']}\n\n"
            f"Consequences:\n{adr['consequences']}"
        )
    except Exception as e:
        return f"Error retrieving ADR: {str(e)}"


@mcp.tool()
def adr_list(status: Optional[str] = None) -> str:
    """
    List all ADRs with optional status filter.
    
    Args:
        status: Filter by status (proposed/accepted/deprecated/superseded)
        
    Returns:
        List of ADRs with number, title, status, and date
        
    Example:
        adr_list()  # All ADRs
        adr_list(status="accepted")  # Only accepted ADRs
    """
    try:
        adrs = adr_ops.list_adrs(status)
        if not adrs:
            return "No ADRs found" + (f" with status '{status}'" if status else "")
        
        result = f"Found {len(adrs)} ADR(s)" + (f" with status '{status}'" if status else "") + ":\n\n"
        for adr in adrs:
            result += f"ADR {adr['number']:03d}: {adr['title']} [{adr['status']}] ({adr['date']})\n"
        
        return result
    except Exception as e:
        return f"Error listing ADRs: {str(e)}"


@mcp.tool()
def adr_search(query: str) -> str:
    """
    Full-text search across all ADRs.
    
    Args:
        query: Search query
        
    Returns:
        List of matching ADRs with context snippets
        
    Example:
        adr_search("Protocol 101")
        adr_search("FastAPI")
    """
    try:
        results = adr_ops.search_adrs(query)
        if not results:
            return f"No ADRs found matching '{query}'"
        
        output = f"Found {len(results)} ADR(s) matching '{query}':\n\n"
        for result in results:
            output += f"ADR {result['number']:03d}: {result['title']}\n"
            for match in result['matches']:
                output += f"  - {match}\n"
            output += "\n"
        
        return output
    except Exception as e:
        return f"Error searching ADRs: {str(e)}"


if __name__ == "__main__":
    mcp.run()
