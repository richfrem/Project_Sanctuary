# ADR MCP Server Documentation

## Overview

ADR MCP manages Architecture Decision Records (ADRs) for Project Sanctuary. It provides operations to create, retrieve, search, and update ADRs with proper status transitions and validation.

## Key Concepts

- **Status Transitions:** Proposed → Accepted → Deprecated/Superseded
- **Validation:** Enforces ADR structure and status transition rules
- **Automatic Numbering:** Sequential ADR numbering
- **Search:** Full-text search across all ADRs

## Server Implementation

- **Server Code:** [mcp_servers/adr/server.py](../../../mcp_servers/adr/server.py)
- **Operations:** [mcp_servers/adr/operations.py](../../../mcp_servers/adr/operations.py)
- **Validator:** [mcp_servers/adr/validator.py](../../../mcp_servers/adr/validator.py)
- **Models:** [mcp_servers/adr/models.py](../../../mcp_servers/adr/models.py)

## Testing

- **Test Suite:** [tests/mcp_servers/adr/](../../../tests/mcp_servers/adr/)
- **Status:** ✅ 14/14 tests passing

## Operations

### `adr_create`
Create a new ADR with automatic sequential numbering

**Example:**
```python
adr_create(
    title="Separation of Council MCP and Agent Persona MCP",
    context="Project Sanctuary implements a 12-domain MCP architecture...",
    decision="We will maintain Council MCP and Agent Persona MCP as separate...",
    consequences="Positive: Modularity, Scalability. Negative: Complexity...",
    status="proposed",
    author="AI Assistant"
)
```

### `adr_get`
Retrieve a specific ADR by number

### `adr_list`
List all ADRs with optional status filter

### `adr_search`
Full-text search across all ADRs

### `adr_update_status`
Update the status of an existing ADR

**Valid Transitions:**
- proposed → accepted
- proposed → deprecated
- accepted → deprecated
- accepted → superseded

## Directory Structure

```
ADRs/
├── 001_initial_architecture.md
├── 042_separation_of_council_mcp_and_agent_persona_mcp.md
└── ...
```

## Status

✅ **Fully Operational** - All operations tested and working
