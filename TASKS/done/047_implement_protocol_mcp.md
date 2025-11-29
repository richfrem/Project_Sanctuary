# Task #047: Implement Protocol MCP

**Status:** Done  
**Priority:** High  
**Estimated Effort:** 3-4 days  
**Dependencies:** Shared Infrastructure  
**Domain:** `project_sanctuary.system.protocol`

---

## Objective

Implement Protocol MCP server for managing system protocols in `01_PROTOCOLS/`.

---

## Key Features

### Tool Signatures

```typescript
protocol_create(
  number: number,
  title: string,
  status: "proposed" | "canonical" | "deprecated",
  classification: string,
  version: string,
  authority: string,
  content: string,
  linked_protocols?: string
): FileOperationResult

protocol_update(
  number: number,
  updates: Partial<Protocol>,
  reason: string
): FileOperationResult

protocol_get(number: number): Protocol

protocol_list(status?: string): QueryResult<Protocol>

protocol_search(query: string): QueryResult<Protocol>
```

---

## Safety Rules

- **Protocol numbers**: Can be manually assigned (unlike ADRs/Chronicles) but must be unique.
- **Canonical Status**: Protocols marked `CANONICAL` require strict version control.
- **Format**: Must follow the standard Protocol header format.
- **P101 compliance**: Auto-generates commit manifest.

---

## Implementation Checklist

### Phase 1: Core Structure
- [ ] Create `mcp_servers/system/protocol/` directory
- [ ] Implement `Protocol` models and schema
- [ ] Implement `validator.py` (header validation, uniqueness)

### Phase 2: Tool Implementation
- [ ] Implement `create_protocol()`
- [ ] Implement `update_protocol()`
- [ ] Implement `get_protocol()`
- [ ] Implement `list_protocols()`
- [ ] Implement `search_protocols()`

### Phase 3: MCP Server
- [ ] Create `server.py`
- [ ] Create `Dockerfile`
- [ ] Create `requirements.txt`

### Phase 4: Testing
- [ ] Unit tests for validator and operations
- [ ] End-to-end workflow tests

---

**Created:** 2025-11-28  
**Domain:** `project_sanctuary.system.protocol`  
**Class:** `project_sanctuary_system_protocol`
