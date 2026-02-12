# Mission: Implement `TaskStatusUpdater` in `verify_inner_loop_result.py`
**(Strategy Packet for Inner Loop / Opus)**

> **Objective:** Execute the task defined below. This packet is your entire context.

## 1. Context
- **Spec**: `[not provided]`
- **Plan**: `[not provided]`
- **Goal**: Implement `TaskStatusUpdater` in `verify_inner_loop_result.py`

## 2. Tasks

- **Task**: Add logic to find a task by ID and check the box in `tasks.md`.
  - **CLI**: Add `--update-status config/tasks.md` flag.
  - **Acceptance**: Running verify on a passing result automatically marks the task `[x]`.

## 3. Constraints
- **NO GIT COMMANDS**: The Outer Loop handles all version control.
- **Token Efficiency**: Produce only the requested artifacts, nothing extra.
- **File Paths**: Use exact paths as specified in the task.

## 4. Acceptance Criteria
- [ ] All files specified in section 2 exist and are correctly implemented.
- [ ] No git commands were executed.
- [ ] Code follows project coding conventions.
