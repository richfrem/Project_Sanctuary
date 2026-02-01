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
