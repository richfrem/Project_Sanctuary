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

- [x] **Create `run_workflow.py` (Pure Python Orchestrator)** <!-- id: 3 -->
  - **Task**: Python script that chains the commands (replacing .sh).
  - **Input**: Task ID.
  - **Logic**:
    1. Call internal generation tool.
    2. Print the `claude` launch command.
    3. (Optional) Wait for user signal to run verification.
  - **Goal**: One-click experience.

- [x] **Update Documentation** <!-- id: 4 -->
  - **Task**: Update `tools/orchestrator/dual_loop/README.md`.
  - **Task**: Update `SKILL.md` to reflect new workflow.

---
# Meta-Tasks (Automated Checklist)

## Learning Loop (Protocol 128)
- [x] **Read Boot Contract & Primer**
- [x] **Review Learning Snapshot**
- [x] **Check Tool RLM Cache**
- [x] **Identify New Tools/Skills**
- [x] **Code Audit** (Pure Python Verified)
- [x] **Distill RLM Cache** (Tools Added)
- [x] **Run Retrospective**
- [x] **Update Learning Handoff**
- [x] **Seal Session**
- [x] **Persist Soul**

## Dual-Loop (Protocol 133)
- [x] **Generate Strategy Packet**
- [x] **Launch Inner Loop**
- [x] **Monitor Inner Loop**
- [x] **Verify Result**
- [x] **Update Task Status**
- [x] **Review Snapshot**

## Spec Kitty Workflow
- [x] **Check Prerequisites**
- [x] **Plan Workflow**
- [x] **Generate Tasks**
- [x] **Visualize Status**
- [x] **Review Completed WPs**
- [x] **Move to Review**
- [ ] **Final Acceptance**
- [ ] **Merge Feature**
