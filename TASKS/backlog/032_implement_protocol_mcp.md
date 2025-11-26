# Task #032: Implement Protocol MCP

**Status:** Backlog  
**Priority:** High  
**Estimated Effort:** 3-4 days  
**Dependencies:** Task #028, Shared Infrastructure  
**Domain:** `project_sanctuary.document.protocol`

---

## Objective

Implement Protocol MCP server for governing rules in `01_PROTOCOLS/`.

---

## Key Features

```typescript
create_protocol(number, title, classification, content, status?, version?, linked_protocols?)
update_protocol(number, updates, changelog)
get_protocol(number)
list_protocols(classification?, status?)
search_protocols(query)
archive_protocol(number, reason)
```

---

## Safety Rules

- Protocol numbers must be unique
- Cannot delete protocols (archive only)
- Updates to canonical protocols require version bump
- Must include changelog for updates
- Protected protocols require explicit approval

---

**Domain:** `project_sanctuary.document.protocol`  
**Class:** `project_sanctuary_document_protocol`
