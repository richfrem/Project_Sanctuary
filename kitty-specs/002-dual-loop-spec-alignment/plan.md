# Plan: Dual-Loop / Spec Kitty Integration

**Feature**: 002-dual-loop-spec-alignment
**Goal**: Integrate Dual-Loop tools with Spec Kitty workflow.

## Phase 1: Robust Task Parsing
**Objective**: Replace regex parsing with a structured markdown parser that handles Spec Kitty's `tasks.md` format (including nested lists and checkboxes).

- [ ] **Refactor `generate_strategy_packet.py`**:
    - Implement a `TaskParser` class.
    - Support finding tasks by `WP-NN` ID or explicit `<!-- id: NN -->` comments if used.
    - extract the full body of the task, including sub-checkboxes.

## Phase 2: Kanban Automation
**Objective**: Enable the verifier to "check off" tasks in the `tasks.md` file.

- [ ] **Update `verify_inner_loop_result.py`**:
    - Add `--update-tasks-file <path>` argument.
    - Add `--task-id <id>` argument.
    - Logic: Find the task in `tasks.md` and replace `- [ ]` with `- [x]`.

## Phase 3: Workflow Integration
**Objective**: Create a seamless "Spec -> Packet" command.

- [ ] **Create `spec-kitty-dual-loop.sh`**:
    - A wrapper script that:
        1.  Calls `/spec-kitty.implement`.
        2.  Calls `generate_strategy_packet.py`.
        3.  Prints the `claude "..."` launch command for the user.
- [ ] **Update `SKILL.md`**: Document the new automated workflow.

## 4. Verification Strategy
- **Self-Hosting**: Use the new parser to parse *this* plan's `tasks.md` file.
- **End-to-End Test**: Run the full flow on a dummy feature.
