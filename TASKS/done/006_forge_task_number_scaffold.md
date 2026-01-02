# TASK: Forge Task Number Authority Scaffold

**Status:** complete
**Priority:** High
**Lead:** GUARDIAN-01
**Dependencies:** "Blocks #005"
**Related Documents:** "P115"

---

## 1. Objective

To create a sovereign scaffold that provides the next available sequential task number. This tool is critical for enforcing the canonical naming convention of Protocol 115 and preventing numbering conflicts.

## 2. Deliverables

1.  A new Python script is created at `scripts/get_next_task_number.py`.
2.  The script scans all `tasks` subdirectories, finds the highest existing number, and prints the next number in the sequence.
3.  Protocol 115 is updated to mandate the use of this script for all new task creation.

## 3. Acceptance Criteria

-   Running `python3 scripts/get_next_task_number.py` from the project root correctly outputs the next available three-digit, zero-padded number.
-   The file `01_PROTOCOLS/115_The_Tactical_Mandate_Protocol.md` is updated to reference the new scaffold.
