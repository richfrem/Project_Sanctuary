# Mission: Create `spec-kitty-dual-loop.sh` Wrapper
**(Strategy Packet for Inner Loop / Opus)**

> **Objective:** Execute the task defined below. This packet is your entire context.

## 1. Context
- **Spec**: `[not provided]`
- **Plan**: `[not provided]`
- **Goal**: Create `spec-kitty-dual-loop.sh` Wrapper

## 2. Tasks

- **Task**: Shell script that chains the commands.
  - **Input**: Task ID (e.g., "3").
  - **Step 1**: Run `spec-kitty agent workflow implement --task-id <ID>`.
  - **Step 2**: Extract the worktree directory from the output (it prints `cd .worktrees/...`).
  - **Step 3**: Generate the Strategy Packet (using `generate_strategy_packet.py`).
  - **Step 4**: Launch Claude *inside* the worktree, feeding it the Strategy Packet.
  - **Goal**: One-click experience that respects the Spec Kitty isolation model.

## 3. Constraints
- **NO GIT COMMANDS**: The Outer Loop handles all version control.
- **Token Efficiency**: Produce only the requested artifacts, nothing extra.
- **File Paths**: Use exact paths as specified in the task.

## 4. Acceptance Criteria
- [ ] All files specified in section 2 exist and are correctly implemented.
- [ ] No git commands were executed.
- [ ] Code follows project coding conventions.
