# Implementation Plan: Verify Workflow State Test

**Branch**: `feat/004-verify-workflow-test` | **Date**: 2026-02-12 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/004-verify-workflow-test/spec.md`

## Summary

Implement a unit test suite for `tools/orchestrator/verify_workflow_state.py` to ensure it correctly validates workflow artifacts (spec, plan, tasks, worktrees) and handles edge cases. Refactor the tool if necessary to improve robustness (e.g., path handling).

## Technical Context

**Language/Version**: Python 3.12 (inferred)
**Primary Dependencies**: `unittest` (standard lib) or `pytest`. `unittest.mock` for filesystem mocking.
**Project Type**: Automation Script
**Constraints**: Must run in the existing environment without new heavy dependencies.

## Architecture Decisions

### Problem / Solution
- **Problem**: `verify_workflow_state.py` path resolution is fragile (assumes CWD). Logic for `review` phase is incomplete.
- **Solution**:
    1.  Create `tests/unit/tools/orchestrator/test_verify_workflow_state.py` using `unittest` and `tempfile`/`pathlib` to create dummy workflow states.
    2.  Refactor `verify_workflow_state.py` to accept a `root_dir` argument or reliably detect project root.
    3.  Implement missing `review` phase logic.

### Design Patterns
- **Dependency Injection**: Pass `root_path` to verification functions to allow testing with temporary directories.

## Project Structure

### Source Code

```text
tools/orchestrator/
└── verify_workflow_state.py  # Existing, to be modified

tests/unit/tools/orchestrator/
└── test_verify_workflow_state.py  # New test file
```

## Verification Plan

### Automated Tests
- [ ] Unit Tests: `python3 -m unittest tests/unit/tools/orchestrator/test_verify_workflow_state.py`
    - Test Case 1: Detect missing `kitty-specs`
    - Test Case 2: Detect missing `spec.md` (Specify phase)
    - Test Case 3: Detect missing `plan.md` (Plan phase)
    - Test Case 4: Detect missing `tasks.md` / `WP-*.md` (Tasks phase)
    - Test Case 5: Detect missing/invalid worktree (Implement phase)
    - Test Case 6: Success paths for all phases

### Manual Verification
- [ ] Run the tool against the current feature (`004-verify-workflow-test`) to verify it passes for `specify` and `plan` phases (after we create them).

## Complexity Tracking

N/A
