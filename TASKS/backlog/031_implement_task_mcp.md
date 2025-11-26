# Task #031: Implement Task MCP

**Status:** Backlog  
**Priority:** High  
**Estimated Effort:** 3-4 days  
**Dependencies:** Task #028, Shared Infrastructure  
**Domain:** `project_sanctuary.document.task`

---

## Objective

Implement Task MCP server for workflow management in `TASKS/`.

---

## Key Features

```typescript
create_task(number, title, description, priority, estimated_effort?, dependencies?, status?)
update_task_status(number, new_status, notes?)
update_task(number, updates)
get_task(number)
list_tasks(status?, priority?)
search_tasks(query)
```

---

## Safety Rules

- Task numbers must be unique
- Circular dependency detection
- Status transitions move files between directories
- Cannot delete tasks (archive only)

---

**Domain:** `project_sanctuary.document.task`  
**Class:** `project_sanctuary_document_task`
