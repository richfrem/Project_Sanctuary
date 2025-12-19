# TASK: Create Client Configuration Toggles

**Status:** complete
**Priority:** Medium
**Lead:** Unassigned
**Dependencies:** 122, 123
**Related Documents:** None

---

## 1. Objective

Create clear, separate configuration files for the two operating modes. This gives the user the explicit 'toggle' capability requested to choose between Old Approach (Legacy) and New Approach (Gateway).

## 2. Deliverables

1. mcp_config/legacy_direct.json
2. mcp_config/gateway_routed.json
3. Updated Documentation

## 3. Acceptance Criteria

- mcp_config/legacy_direct.json created and tested with Claude Desktop (simulation)
- mcp_config/gateway_routed.json created targeting Gateway ports
- Instructions added to SIDE_BY_SIDE_PORTS.md

## Notes

**Status Change (2025-12-19):** backlog → in-progress
Starting creation of client configuration files (Task 124).

**Status Change (2025-12-19):** in-progress → complete
Legacy and Gateway configurations created.
