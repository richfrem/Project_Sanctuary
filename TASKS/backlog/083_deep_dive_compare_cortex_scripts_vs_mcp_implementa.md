# TASK: Migrate and Archive Legacy Mnemonic Cortex

**Status:** backlog
**Priority:** Medium
**Lead:** Unassigned
**Dependencies:** None
**Related Documents:** TASKS/done/080_migrate_and_archive_legacy_council_orchestrator.md

---

## 1. Objective

Migrate the legacy `mnemonic_cortex` architecture to the new MCP-first architecture (`mcp_servers/cognitive/cortex`). This involves refactoring the MCP implementation to match the proven logic of the legacy scripts, moving documentation and tests to their new standard locations, and archiving the legacy folder.

## 2. Deliverables

1.  **Refactored Cortex MCP:** `CortexOperations` updated to match `ingest.py` logic (batching, error handling).
2.  **Migrated Documentation:** Cortex docs moved to `docs/mcp/cortex/`.
3.  **Migrated Tests:** Tests moved to `tests/mcp_servers/cortex/` and updated.
4.  **Archived Legacy Code:** `mnemonic_cortex/` moved to `ARCHIVE/mnemonic_cortex/`.
5.  **Updated Operations Inventory:** Reflecting the new structure.

## 3. Implementation Plan

### Phase 1: Code Refactoring & Parity
- [ ] Analyze `mnemonic_cortex/scripts/ingest.py` vs `mcp_servers/lib/cortex/operations.py`.
- [ ] Port "Disciplined Batch Architecture" and recursive retry logic to `CortexOperations`.
- [ ] Verify `cortex_ingest_full` matches legacy script performance and reliability.

### Phase 2: Asset Migration
- [ ] Move documentation from `mnemonic_cortex/` to `docs/mcp/cortex/`.
- [ ] Move tests to `tests/mcp_servers/cortex/`.
- [ ] Update import paths in tests and server code.

### Phase 3: Verification
- [ ] Run full ingestion via MCP tool.
- [ ] Verify Protocol 101 v3.0 indexing.
- [ ] Run automated test suite.

### Phase 4: Archival
- [ ] Move `mnemonic_cortex` to `ARCHIVE/`.
- [ ] Update `task.md` and `README.md` references.
