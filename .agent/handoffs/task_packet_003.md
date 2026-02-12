# Mission: Refactor Task Parser for List Support
**(Strategy Packet for Inner Loop / Opus)**

> **Objective:** Rewrite the task parsing logic in `generate_strategy_packet.py` to support standard markdown list items with ID comments, replacing the brittle header-based regex.

## 1. Context
- **Spec**: `kitty-specs/002-dual-loop-spec-alignment/spec.md`
- **Tasks File**: `kitty-specs/002-dual-loop-spec-alignment/tasks.md`
- **Target File**: `tools/orchestrator/dual_loop/generate_strategy_packet.py`

## 2. Problem
The current parser expects tasks to be H3 headers (`### A. Title`).
However, the project standard (as seen in the provided Tasks File) is:
```markdown
- [ ] **Task Title** <!-- id: 1 -->
  - Subtask details...
  - More details...
```
The tool currently fails to find any tasks in standard files.

## 3. Tasks

### A. Update `generate_strategy_packet.py`
1.  **Rewrite `parse_tasks_file`**:
    - Iterate line by line or uses a robust regex to find list items starting with `- [ ]` or `- [x]`.
    - Extract the **Task ID** from `<!-- id: ... -->` comments on the same line.
    - Extract the **Title** (text between the checkbox and the comment).
    - Extract the **Body** (indented lines following the task item, until the next non-indented task).
2.  **Update `main`**: Ensure it handles numeric IDs nicely (the new tasks use `id: 1`, `id: 2`).

## 4. Constraints
- **NO GIT COMMANDS**: Standard Inner Loop constraint.
- **Standard Library Only**: Use `re` and `pathlib`.
- **Robustness**: Must handle multi-line bodies (indented sub-bullets).

## 5. Acceptance Criteria
- [ ] `generate_strategy_packet.py --tasks-file kitty-specs/002-dual-loop-spec-alignment/tasks.md --task-id 1 --stdout` successfully prints a packet.
- [ ] The printed packet contains the correct Title and Body from the tasks file.
