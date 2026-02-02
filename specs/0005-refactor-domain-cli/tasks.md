# Tasks: Spec 0005 - Migrate Domain CLI

## Phase 1: Analysis & Infrastructure
- [x] T001 Create `cli_gap_analysis.md`.
- [x] T002 Create spec folder with `spec.md`, `plan.md`, `tasks.md`, `scratchpad.md`.
- [x] T003 Create feature branch `spec/0005-refactor-domain-cli`.
- [x] T003b Create `mcp_server_inventory.md` - comprehensive MCP operations audit.

## Phase 2: CLI Implementation
- [x] T004 Add `chronicle` command cluster to `tools/cli.py` (list, search, get, create, update).
- [x] T005 Add `task` command cluster to `tools/cli.py` (list, get, create, update-status, search, update).
- [x] T006 Add `adr` command cluster to `tools/cli.py` (list, search, get, create, update-status).
- [x] T007 Add `protocol` command cluster to `tools/cli.py` (list, search, get, create, update).
- [x] T007b Add `forge` command cluster to `tools/cli.py` (query, status).

## Phase 3: Verification
- [x] T008 Verify `chronicle` commands work.
- [x] T009 Verify `task` commands work.
- [x] T010 Verify `adr` commands work.
- [x] T011 Verify `protocol` commands work.

## Phase 4: Workflow Updates
- [x] T012 Update `workflow-adr.md` to use `tools/cli.py`.
- [x] T013 Update `workflow-chronicle.md` to use `tools/cli.py`.
- [x] T014 Update `workflow-task.md` to use `tools/cli.py`.

## Phase 5: Cleanup & Closure
- [x] T015 Add deprecation headers to `scripts/domain_cli.py` and `scripts/cortex_cli.py`.
- [ ] T016 Run `/workflow-retrospective`.
- [ ] T017 Run `/workflow-end`.

## Phase 6: Unplanned / Scratchpad
- [x] T018 Create `mcp_server_inventory.md` - comprehensive MCP operations audit.
- [x] T019 Add `forge query/status` commands for direct model interaction.
- [x] T020 Add missing update commands (chronicle, task, protocol, adr update-status).


