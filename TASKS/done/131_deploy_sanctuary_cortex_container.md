# TASK: Deploy Sanctuary Cortex Container

**Status:** complete
**Priority:** High
**Lead:** Unassigned
**Dependencies:** None
**Related Documents:** None

---

## 1. Objective

Deploy the sanctuary-cortex container to the Hybrid Fleet.

## 2. Deliverables

1. Dockerfile
2. docker-compose service entry verified
3. Verification script

## 3. Acceptance Criteria

- Container builds successfully
- Exposes SSE endpoint on correct port (8104)
- Connects to VectorDB (8000) and Ollama (11434)
- Guardian Wakeup verified in container

## Notes

**Status Change (2025-12-19):** backlog → complete
Fleet verified 6/6 online.
