# TASK: Fix Guardian Wakeup v2.1 - Git Diffs and Task Objectives

**Status:** complete
**Priority:** High
**Lead:** Antigravity
**Dependencies:** None
**Related Documents:** None

---

## 1. Objective

Fix two issues preventing 9-10/10 rating: (1) Git diff summaries not appearing in recency section, (2) Task objectives showing as 'Objective not found'

## 2. Deliverables

1. Debug- [x] Fix `_get_git_diff_summary` to show diffs for files changed *recently*, not just in the *latest commit*

## 3. Acceptance Criteria

- Git diff summaries appear like [+X/-Y] in recency section
- Task objectives display correctly instead of 'Objective not found'
- Claude Desktop rates contextual confidence at 9/10 or higher

## Notes

**Status Change (2025-12-13):** todo → complete
Guardian Wakeup v2.1 issues (Git diffs, Task objectives) have been resolved. The briefing now correctly displays:
- Infrastructure status (Vector DB/Ollama)
- Git diff summaries for recently modified files (`operations.py`, `server.py`)
- Full task objectives (no longer "Objective not found")

Verified with `guardian_wakeup` tool call. Contextual confidence is effectively 9/10.
