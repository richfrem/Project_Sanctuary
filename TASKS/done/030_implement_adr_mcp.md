# Task #030: Implement ADR MCP

**Status:** Done
**Priority:** High  
**Estimated Effort:** 2-3 days  
**Dependencies:** Task #028, Shared Infrastructure, **Podman configured**  
**Domain:** `project_sanctuary.document.adr`

---

## Objective

Implement ADR MCP server for Architecture Decision Records in `ADRs/`.

---

## Lessons Learned from Task #031 (Task MCP)

1.  **Phased Approach:** Follow the 4-phase structure (Prerequisites -> Core Modules -> MCP Server -> Testing).
2.  **Containerization:** Use Podman. Ensure `Dockerfile` and `requirements.txt` are created early.
3.  **Testing:**
    -   Unit tests for `operations.py` and `validator.py` are critical.
    -   E2E workflow tests (create -> update -> verify) ensure the system works as a whole.
4.  **Documentation:** `README.md` must include Quick Start and Tool Signatures.
5.  **Safety:** File operations only. No direct Git commits (handled by Git Workflow MCP).
6.  **Validation:** Strict schema validation for inputs and dependency checks (e.g., `supersedes` references existing ADRs).

---

## Key Features

```typescript
create_adr(number, title, context, decision, consequences, date?, status?, supersedes?)
update_adr_status(number, new_status, reason)
get_adr(number)
list_adrs(status?)
search_adrs(query)
```

---

## Safety Rules

- ADR numbers must be sequential
- Cannot delete ADRs (mark as superseded)
- Status transitions must be valid
- Follows ADR template format

---

## Implementation Plan

### Phase 1: Prerequisites & Documentation
- [ ] Create `mcp_servers/adr/README.md`
- [ ] Define `models.py` (ADRSchema, enums)

### Phase 2: Core Modules
- [ ] Create `mcp_servers/adr/__init__.py`
- [ ] Create `validator.py` (schema validation, sequential checks)
- [ ] Create `operations.py` (file operations)

### Phase 3: MCP Server
- [ ] Create `server.py` (MCP protocol implementation)
- [ ] Create `Dockerfile`
- [ ] Create `requirements.txt`
- [ ] Build container image (`adr-mcp:latest`)

### Phase 4: Testing & Validation
- [ ] **Unit Tests**
    - [ ] `test_validator.py`
    - [ ] `test_operations.py`
- [ ] **End-to-End Tests**
    - [ ] `test_e2e_workflow.py` (Create -> Update Status -> Search)
- [ ] **Manual Verification**
    - [ ] Run in Podman
    - [ ] Connect via Inspector or LLM

---

**Domain:** `project_sanctuary.document.adr`  
**Class:** `project_sanctuary_document_adr`
**Port:** 3003 (Planned)
