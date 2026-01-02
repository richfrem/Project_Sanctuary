# TASK: Investigate and Relocate Misplaced core/ Folder

**Status:** todo
**Priority:** Medium
**Lead:** Unassigned
**Dependencies:** None
**Related Documents:** core/, mcp_servers/, docs/architecture/mcp/architecture.md

---

## 1. Objective

Investigate and resolve the misplaced core/ folder in project root. Determine if it should be moved to mcp_servers/shared/ or another appropriate location per the MCP architecture, then refactor imports accordingly.

## 2. Deliverables

1. Investigation report on core/ folder purpose and dependencies
2. Refactoring plan document
3. Move core/ modules to correct location
4. Update import statements project-wide
5. Update .gitignore if needed
6. Verification test run

## 3. Acceptance Criteria

- Determine correct location for core/ modules
- Move core/ to appropriate location (likely mcp_servers/shared/ or lib/)
- Update all import statements across the project
- Verify no broken imports remain
- Update documentation to reflect new structure
- All tests pass after refactoring

## Notes

The core/ folder in project root contains git/ and utils/ subdirectories. This appears to be shared infrastructure that should likely be in mcp_servers/shared/ or a similar location per the MCP architecture. Need to investigate dependencies and determine correct placement.
