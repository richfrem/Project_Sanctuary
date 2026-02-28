# Dual-Loop (Protocol 133) Meta-Tasks
<!-- To be included in Session Task List for any Dual-Loop Execution -->

## Phase A: Strategy (Outer Loop)
- [ ] **Verify Spec Kitty artifacts**: `python3 tools/orchestrator/verify_workflow_state.py --feature <SLUG> --phase tasks`
- [ ] **Create worktree**: `/spec-kitty.implement <WP-ID>` (or use `--skip-worktree` for Branch-Direct)
- [ ] **Generate Strategy Packet**: `python3 tools/orchestrator/dual_loop/generate_strategy_packet.py --tasks-file <PATH> --task-id <WP-ID>`

## Phase B: Hand-off & Execution
- [ ] **Hand off to Inner Loop**: `claude "Read .agent/handoffs/task_packet_NNN.md. Execute the mission. Do NOT use git."`
- [ ] **Inner Loop completes**: All acceptance criteria met, no git commands used

## Phase C: Verification (Outer Loop)
- [ ] **Verify result**: `python3 tools/orchestrator/dual_loop/verify_inner_loop_result.py --packet <PATH> --verbose`
- [ ] **Verify clean state**: `python3 tools/orchestrator/verify_workflow_state.py --wp <WP-ID> --phase review`
- [ ] **On PASS**: Commit in worktree, update task lane to `done`
- [ ] **On FAIL**: Hand off `correction_packet_NNN.md`, repeat Phase B

## Phase D: Closure (Protocol 128)
- [ ] **Seal**: `/sanctuary-seal`
- [ ] **Persist**: `/sanctuary-persist`
- [ ] **Retrospective**: `/sanctuary-retrospective`
- [ ] **End**: `/sanctuary-end`
