# TASK: Implement ADR 063 Structural Segregation

**Status:** complete
**Priority:** Medium
**Lead:** Unassigned
**Dependencies:** None
**Related Documents:** None

---

## 1. Objective

Implement the Total Mirror Architecture defined in ADR 063 v1.0, enforcing strict namespacing for the Fleet of 8 to avoid collision with Legacy 12 servers.

## 2. Deliverables

1. Restructured filesystem
2. Updated docker-compose.yml
3. Cleaned up legacy files

## 3. Acceptance Criteria

- Directory structure in mcp_servers matches docs/mcp_servers/gateway/clusters/ (kebab-case).
- docker-compose.yml build contexts point to new locations.
- Legacy top-level folders (utils, git, etc) are removed.
- System passes 'The Phoenix Test' (make up / make status).

## Notes

**Status Change (2025-12-20):** in-progress → complete
Renaming complete. Structure verified. Legacy files preserved.
