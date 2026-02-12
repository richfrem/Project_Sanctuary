# Feature Specification: Verify Workflow State Test

**Feature Branch**: `004-verify-workflow-test`  
**Category**: Process
**Created**: 2026-02-12
**Status**: Draft  
**Input**: User description: "Verify tools/orchestrator/verify_workflow_state.py within a clean Dual-Loop execution"

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Verify Feature Phase Integrity (Priority: P1)

As an Agent or Developer, I want to verify that all necessary planning artifacts exist before proceeding to the next phase, so that I don't build on missing foundations.

**Why this priority**: Essential for the `verify_workflow_state.py` tool to prevent broken workflows.

**Independent Test**: Can be tested by creating dummy features with missing files and asserting the tool fails.

**Acceptance Scenarios**:

1. **Given** a feature with no `spec.md`, **When** I run verification for `specify` phase, **Then** it fails with "Missing spec.md".
2. **Given** a feature with `spec.md` but no `plan.md`, **When** I run verification for `plan` phase, **Then** it fails with "Missing plan.md".
3. **Given** a complete feature, **When** I run verification for `tasks` phase, **Then** it succeeds.

---

### User Story 2 - Verify Implementation Phase Integrity (Priority: P1)

As an Agent, I want to ensure the worktree is correctly set up before starting coding, so that I don't edit the wrong files.

**Why this priority**: Critical for worktree isolation.

**Independent Test**: Run tool against non-existent WP ID.

**Acceptance Scenarios**:

1. **Given** a WP ID that hasn't been implemented, **When** I run verify for `implement`, **Then** it fails with "Worktree not found".
2. **Given** a valid worktree, **When** I run verify for `implement`, **Then** it passes.

---

### Edge Cases

- What happens when the feature directory exists but is empty? -> Should fail specific file checks.
- How does system handle malformed arguments? -> Argparse should handle it.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: Tool MUST accept `--feature` and `--phase` arguments to validate planning artifacts.
- **FR-002**: Tool MUST accept `--wp` and `--phase` arguments to validate implementation state.
- **FR-003**: Tool MUST return non-zero exit code on failure.
- **FR-004**: Tool MUST print distinct success/failure messages to stdout/stderr.
- **FR-005**: Tool MUST correctly locate `kitty-specs` directory from project root.
- **FR-006**: Tool MUST correctly locate `.worktrees` directory from project root.

### Key Entities

- **Feature Directory**: `kitty-specs/<slug>`
- **Worktree**: `.worktrees/<slug>-<WP>`

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Tool correctly identifies 100% of missing artifact scenarios defined in test cases.
- **SC-002**: Tool completes verification in under 1 second.
- **SC-003**: Tool usage is documented and matches implementation.
