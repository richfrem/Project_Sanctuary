# Protocol MCP Server Documentation

## Overview

Protocol MCP manages the protocol documents in the `01_PROTOCOLS/` directory. It provides operations to create, retrieve, search, and update protocols with proper validation.

## Key Concepts

- **Status Management:** PROPOSED → CANONICAL → DEPRECATED
- **Classification:** Foundational Framework, Operational Procedure, etc.
- **Version Control:** Semantic versioning for protocols
- **Linked Protocols:** Cross-references between related protocols

## Server Implementation

- **Server Code:** [mcp_servers/protocol/server.py](../../../mcp_servers/protocol/server.py)
- **Operations:** [mcp_servers/protocol/operations.py](../../../mcp_servers/protocol/operations.py)
- **Validator:** [mcp_servers/protocol/validator.py](../../../mcp_servers/protocol/validator.py)
- **Models:** [mcp_servers/protocol/models.py](../../../mcp_servers/protocol/models.py)

## Testing

- **Test Suite:** [tests/mcp_servers/protocol/](../../../tests/mcp_servers/protocol/)
- **Status:** ✅ 14/14 tests passing

## Operations

### `protocol_create`
Create a new protocol

**Example:**
```python
protocol_create(
    number=120,
    title="MCP Composition Patterns",
    status="PROPOSED",
    classification="Foundational Framework",
    version="1.0",
    authority="AI Assistant",
    content="# Protocol 120: MCP Composition Patterns\n\n...",
    linked_protocols=["101", "115"]
)
```

### `protocol_get`
Retrieve a specific protocol by number

### `protocol_list`
List protocols with optional status filter

### `protocol_search`
Full-text search across all protocols

### `protocol_update`
Update an existing protocol

## Directory Structure

```
01_PROTOCOLS/
├── 101_project_sanctuary_foundation.md
├── 120_mcp_composition_patterns.md
└── ...
```

## Status

✅ **Fully Operational** - All operations tested and working
