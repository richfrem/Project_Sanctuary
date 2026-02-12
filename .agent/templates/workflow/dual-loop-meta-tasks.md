# Dual-Loop (Protocol 133) Meta-Tasks
<!-- To be included in Session Task List for any Dual-Loop Execution -->

## Phase A: Outer Loop (Orchestration)
- [ ] **Generate Strategy Packet**: `python3 tools/orchestrator/dual_loop/generate_strategy_packet.py`
- [ ] **Launch Inner Loop**: `python3 tools/orchestrator/dual_loop/run_workflow.py <TASK_ID>`
- [ ] **Monitor Inner Loop**: Check console for tool usage / progress.

## Phase B: Verification (Outer Loop)
- [ ] **Verify Result**: `python3 tools/orchestrator/dual_loop/verify_inner_loop_result.py`
- [ ] **Update Task Status**: Ensure `--update-status` flag marked task as complete.
- [ ] **Review Snapshot**: Check generated artifacts (diffs, reports).
