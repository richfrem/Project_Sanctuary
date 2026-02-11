# Tasks: Spec 0004 - Standardize CLI

## Phase 1: Analysis & Infrastructure
- [x] T001 Analyze `scripts/cortex_cli.py` vs `tools/cli.py`.
- [x] T002 Create `cli_gap_analysis.md`.
- [x] T003 Update `spec.md` and `plan.md` with revised approach.

## Phase 2: CLI Implementation
- [x] T004 Update `tools/cli.py`: Add `--output` to `debrief` command.
- [x] T005 Update `tools/cli.py`: Add `bootstrap-debrief` command.
- [x] T006 Update `tools/cli.py`: Add `--manifest` to `guardian` command.
- [x] T007 Verify imports are strictly from `mcp_servers` (no missing dependencies).

## Phase 3: Verification
- [x] T008 Verify `debrief --output`.
- [x] T009 Verify `bootstrap-debrief`.
- [x] T010 Verify `guardian wakeup`.

## Phase 4: Closure
- [ ] T011 Run `/sanctuary-retrospective`.
- [ ] T012 Run `/sanctuary-end`.

## Phase 5: Unplanned / Scratchpad
- [x] T013 Create Backlog Task #162 (Domain CLI migration).
- [x] T014 Consolidate Snapshot Tools (Delete JS, Keep Python, Update Inventories).
