# Tasks: Dual-Loop / Spec Kitty Integration

**Feature**: `002-dual-loop-spec-alignment`
**Workflow**: `/spec-kitty.tasks`

## Phase 1: Robust Tooling

- [x] **Refactor `generate_strategy_packet.py` for List-Based Parsing** <!-- id: 1 -->
  - **Context**: Current implementation expects `### ID` headers, but standard is `- [ ] ... <!-- id: N -->`.
  - **Task**: Rewrite `parse_tasks_file` to handle list items with `<!-- id: ... -->` comments.
  - **Constraint**: Must support multi-line task bodies (indented contents).
  - **Output**: Valid strategy packet from a standard `tasks.md` list item.

- [x] **Implement `TaskStatusUpdater` in `verify_inner_loop_result.py`** <!-- id: 2 -->
  - **Task**: Add logic to find a task by ID and check the box in `tasks.md`.
  - **CLI**: Add `--update-status config/tasks.md` flag.
  - **Acceptance**: Running verify on a passing result automatically marks the task `[x]`.

## Phase 2: Workflow Integration

- [ ] **Create `spec-kitty-dual-loop.sh` Wrapper** <!-- id: 3 -->
  - **Task**: Shell script that chains the commands.
  - **Input**: Task ID.
  - **Logic**:
    1. Call internal generation tool.
    2. Print the `claude` launch command.
    3. (Optional) Wait for user signal to run verification.
  - **Goal**: One-click experience.

- [ ] **Update Documentation** <!-- id: 4 -->
  - **Task**: Update `tools/orchestrator/dual_loop/README.md`.
  - **Task**: Update `SKILL.md` to reflect new workflow.
