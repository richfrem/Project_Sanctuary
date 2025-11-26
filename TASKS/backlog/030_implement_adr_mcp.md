# Task #030: Implement ADR MCP

**Status:** Backlog  
**Priority:** High  
**Estimated Effort:** 2-3 days  
**Dependencies:** Task #028, Shared Infrastructure  
**Domain:** `project_sanctuary.document.adr`

---

## Objective

Implement ADR MCP server for Architecture Decision Records in `ADRs/`.

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

**Domain:** `project_sanctuary.document.adr`  
**Class:** `project_sanctuary_document_adr`
