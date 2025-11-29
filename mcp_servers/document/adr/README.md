# ADR MCP Server

**Domain:** `project_sanctuary.document.adr`  
**Version:** 1.0.0  
**Status:** Production Ready

---

## Overview

The ADR MCP server provides tools for managing Architecture Decision Records (ADRs) in the `ADRs/` directory. It enforces the canonical ADR schema, validates sequential numbering, and provides search capabilities.

**Key Principle:** Safe, validated ADR management with no git operations.

---

## Quick Start

### Prerequisites

1. **Python 3.11+**
2. **Project Sanctuary** repository

### Start the MCP Server

**Local Development:**
```bash
cd /Users/richardfremmerlid/Projects/Project_Sanctuary
python3 -m mcp_servers.document.adr.server
```

**Via Claude Desktop / Antigravity:**
Already configured in MCP config. Just restart the client.

---

## Tools (5)

### 1. `adr_create`
Create a new ADR with automatic sequential numbering.

**Arguments:**
- `title` (str): ADR title
- `context` (str): Problem description and background
- `decision` (str): What was decided and why
- `consequences` (str): Positive/negative outcomes
- `date` (str, optional): Decision date (defaults to today)
- `status` (str, optional): Initial status (defaults to "proposed")
- `author` (str, optional): Decision maker (defaults to "AI Assistant")
- `supersedes` (int, optional): ADR number this supersedes

**Returns:**
```json
{
  "adr_number": 38,
  "file_path": "ADRs/038_example_decision.md",
  "status": "proposed"
}
```

**Example:**
```python
adr_create(
    title="Adopt FastAPI for REST APIs",
    context="Need a modern Python web framework...",
    decision="We will use FastAPI for all REST APIs...",
    consequences="Positive: Fast, modern, async support..."
)
```

---

### 2. `adr_update_status`
Update the status of an existing ADR.

**Arguments:**
- `number` (int): ADR number
- `new_status` (str): New status (proposed/accepted/deprecated/superseded)
- `reason` (str): Reason for status change

**Valid Transitions:**
- proposed → accepted
- proposed → deprecated
- accepted → deprecated
- accepted → superseded

**Returns:**
```json
{
  "adr_number": 38,
  "old_status": "proposed",
  "new_status": "accepted",
  "updated_at": "2025-11-27"
}
```

---

### 3. `adr_get`
Retrieve a specific ADR by number.

**Arguments:**
- `number` (int): ADR number

**Returns:**
```json
{
  "number": 38,
  "title": "Adopt FastAPI for REST APIs",
  "status": "accepted",
  "date": "2025-11-27",
  "author": "AI Assistant",
  "context": "...",
  "decision": "...",
  "consequences": "..."
}
```

---

### 4. `adr_list`
List all ADRs with optional status filter.

**Arguments:**
- `status` (str, optional): Filter by status

**Returns:**
```json
{
  "adrs": [
    {
      "number": 37,
      "title": "MCP Git Migration Strategy",
      "status": "accepted",
      "date": "2025-11-27"
    },
    ...
  ]
}
```

---

### 5. `adr_search`
Full-text search across all ADRs.

**Arguments:**
- `query` (str): Search query

**Returns:**
```json
{
  "results": [
    {
      "number": 37,
      "title": "MCP Git Migration Strategy",
      "matches": [
        "...Protocol 101...",
        "...Smart Git MCP..."
      ]
    }
  ]
}
```

---

## Safety Rules

1. **Sequential Numbering**: ADR numbers are automatically assigned sequentially
2. **No Deletion**: ADRs cannot be deleted, only superseded
3. **Valid Transitions**: Status changes must follow allowed transitions
4. **Supersedes Validation**: Referenced ADRs must exist
5. **Schema Compliance**: All ADRs follow the canonical schema
6. **File Operations Only**: No git commits (use Git Workflow MCP)

---

## ADR Schema

All ADRs follow this format:

```markdown
# [Decision Title]

**Status:** [proposed | accepted | deprecated | superseded]
**Date:** YYYY-MM-DD
**Author:** [Name]
**Context:** [Optional task reference]

---

## Context

[Problem description and background]

## Decision

[What was decided and why]

## Consequences

### Positive
- [Benefits]

### Negative
- [Trade-offs]

### Risks
- [Potential issues and mitigation]
```

---

## Configuration

### Claude Desktop / Antigravity
```json
{
  "adr": {
    "displayName": "ADR MCP",
    "command": "/usr/local/bin/python3",
    "args": ["-m", "mcp_servers.document.adr.server"],
    "env": {
      "PROJECT_ROOT": "/Users/richardfremmerlid/Projects/Project_Sanctuary",
      "PYTHONPATH": "/Users/richardfremmerlid/Projects/Project_Sanctuary"
    },
    "cwd": "/Users/richardfremmerlid/Projects/Project_Sanctuary"
  }
}
```

---

## Testing

```bash
# Run all tests
PYTHONPATH=. python3 -m pytest tests/test_adr*.py -v

# Test specific functionality
PYTHONPATH=. python3 -m pytest tests/test_adr_operations.py::test_create_adr -v
```

---

## Troubleshooting

### ADR Number Already Exists
```
Error: ADR 038 already exists
```
**Solution:** The system automatically assigns the next available number.

### Invalid Status Transition
```
Error: Cannot transition from 'accepted' to 'proposed'
```
**Solution:** Check valid transitions in the Safety Rules section.

### Superseded ADR Not Found
```
Error: ADR 025 does not exist (referenced in supersedes)
```
**Solution:** Ensure the referenced ADR exists before creating a superseding ADR.

---

## Related Documentation

- [ADR Schema](../../ADRs/adr_schema.md)
- [MCP Architecture](../../docs/mcp/architecture.md)
- [Task MCP](../task/README.md)

---

**Last Updated:** 2025-11-27  
**Maintainer:** Project Sanctuary Team
