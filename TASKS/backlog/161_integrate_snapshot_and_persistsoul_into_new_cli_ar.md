# TASK: Integrate Snapshot and Persist-Soul into New CLI Architecture

**Status:** backlog
**Priority:** Medium
**Lead:** Unassigned
**Dependencies:** None
**Related Documents:** None

---

## 1. Objective

Create workflows and CLI integration for snapshot (seal/audit) and persist-soul commands that align with the new bundling CLI architecture. New cli.py should call functions from existing cortex_cli.py/operations.py

## 2. Deliverables

1. workflow-snapshot.md
2. workflow-persist.md updated
3. New cli.py integration with snapshot/persist-soul

## 3. Acceptance Criteria

- snapshot --type seal generates learning_package_snapshot.md
- persist-soul uploads to HuggingFace
- Integrated with /workflow-seal and /workflow-persist

## 4. Implementation Guide (Added from Code Analysis)

**Source Module:** `mcp_servers.learning.operations.LearningOperations`
- **Location:** `mcp_servers/learning/operations.py`
- **Protocol:** Protocol 128 (Learning Loop)

**Snapshot Command (`snapshot --type`)**:
1.  **Operation:** `ops.capture_snapshot(snapshot_type=...)`
2.  **Types:**
    - `audit`: For Red Team review (generates `learning_audit_packet`)
    - `seal`: For session close (generates `learning_package_snapshot.md`)
3.  **Validation:** `mcp_servers.learning.validator.LearningValidator.validate_snapshot_type`

**Persist Soul Command (`persist-soul`)**:
1.  **Operation:** `ops.persist_soul()`
2.  **Functionality:**
    - Broadcasts learnings to Hugging Face
    - Updates `lineage/` with markdown snapshots
    - Syncs `data/soul_traces.jsonl`
3.  **Requirements:** Must strictly follow "Protocol 128: Pre-Departure Checklist".

**CLI Structure:**
- Should reuse `mcp_servers.lib.path_utils.find_project_root()` standard.
- Should leverage `mcp_servers.lib.logging_utils` for consistent output format.
