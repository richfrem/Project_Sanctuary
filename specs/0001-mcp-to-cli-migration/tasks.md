---
description: "Task list for MCP-to-CLI Migration (Phase 1: Foundation)"
---

# Tasks: MCP-to-CLI Migration

**Input**: Design documents from `/specs/0001-mcp-to-cli-migration/`
**Prerequisites**: plan.md ✅, spec.md ✅

## Naming Conventions
- **Task ID**: T001, T002... (Sequential)
- **Story Label**: [US1], [US2]... (Mandatory for story tasks)
- **Parallel**: [P] (Optional, indicates no dependencies)

---

## Phase 0: Pre-Flight ✅
- [x] T000 Verify on `main` branch
- [x] T001 Create `specs/` directory structure
- [x] T002 Initialize Spec Bundle (`spec.md`, `plan.md`, `scratchpad.md`)

## Phase 1: Fix Broken Tools
- [x] T003 Fix `next_number.py` SyntaxWarning (escape sequence on line 169)
- [x] T004 Fix `next_number.py` ADR directory path (`ADRs` not `docs/ADRs`)
- [x] T005 Fix `query_cache.py` BrokenPipeError when piping to `head`
- [x] T006 [P] Update `workflow-start.md` and `WorkflowManager.py` to use `--type` flag for next_number.py

## Phase 2: Verify Foundation [US2]
- [x] T007 [US2] Test `WorkflowManager.start_workflow()` without MCP servers
- [x] T008 [US2] Verify `python tools/cli.py workflow start --name test --target 001` works
- [x] T009 [P] [US2] Document any missing CLI subcommands in cli.py

## Phase 3: Tool Discovery [US1]
- [x] T010 [US1] Verify RLM cache contains tool summaries
- [x] T011 [US1] Test `query_cache.py` returns results for "workflow" query
- [x] T012 [US1] Test `query_cache.py` returns results for "chronicle" query
- [x] T013 [P] [US1] Register new tools in inventory if missing

## Phase 4: Documentation [US3]
- [x] T014 [US3] Create `docs/operations/mcp/mcp_to_cli_mapping.md` skeleton (See `operations_matrix.md`)
- [x] T015 [US3] Map Chronicle MCP operations to CLI equivalents (`domain_cli.py`)
- [x] T016 [US3] Map Git MCP operations to CLI equivalents (Implicit in Workflows)
- [x] T017 [P] [US3] Map remaining high-priority MCPs (Protocol, ADR, Task) (`domain_cli.py`)

## Phase 5: Workflow Migration
- [ ] T018 Update `speckit-analyze.md`: Replace `source scripts/bash/` with `python tools/cli.py`
- [ ] T019 Update `speckit-checklist.md`: Replace `source scripts/bash/` with `python tools/cli.py`
- [ ] T020 Update `speckit-tasks-to-issues.md`: Replace bash pre-flight, address MCP tool dependency
- [ ] T021 Audit remaining speckit workflows for bash/MCP references
- [x] T022 Update `recursive_learning.md`: Replace "Cortex MCP Suite" references with CLI

- [x] T022 Update `recursive_learning.md`: Replace "Cortex MCP Suite" references with CLI
     
## Phase 5b: Protocol 128 Implementation (New)
- [x] T035 Create `/workflow-learning-loop.md` and shim (The Orchestrator)
- [x] T036 Create Atomic Phase workflows (`scout`, `audit`, `seal`, `persist`, `ingest`) and shims
- [x] T037 Create Domain workflows (`chronicle`, `task`, `adr`) and shims
- [x] T038 Update `workflow-composition.mmd` to feature Learning Loop
- [x] T039 Refactor `hybrid-spec-workflow.mmd` to use Learning Loop as Standard Path
- [x] T040 Update `protocol_128_learning_loop.mmd` with workflow refs and context links

## Phase 6: Bash Script Migration (ADR-096) [OUT OF SCOPE]
- [ ] T023 Analyze `update-agent-context.sh` (26KB) → Plan Python equivalent (DEFERRED)
- [ ] T024 Analyze `create-new-feature.sh` (10KB) → Merge into WorkflowManager (DEFERRED)
- [x] T025 Verify thin shims (`workflow-start.sh`, `workflow-end.sh`, `workflow-retrospective.sh`) only contain `exec python3`

## Backlog / Future
- [ ] T041 Create new Spec for ADR-096 Bash Migration (Phase 6 tasks deferred)

## Phase 7: Documentation
- [x] T026 [US3] Create `docs/operations/mcp/mcp_to_cli_mapping.md` skeleton
- [x] T027 [US3] Map Chronicle MCP operations to CLI equivalents
- [x] T028 [US3] Map Git MCP operations to CLI equivalents
- [x] T029 [P] [US3] Update constitution/rules to reflect CLI-first policy

## Phase N: Closure & Merge (MANDATORY)
- [ ] T030 Run `/workflow-retrospective`
- [ ] T031 Commit all changes to Feature Branch
- [ ] T032 Push to remote and Create Pull Request
- [ ] T033 Confirm with User: "PR Merged?"
- [ ] T034 Run `/workflow-end`

---

## Progress Summary

| Phase | Status | Tasks |
|-------|--------|-------|
| Phase 0: Pre-Flight | ✅ Complete | 3/3 |
| Phase 1: Fix Tools | ✅ Complete | 4/4 |
| Phase 2: Verify Foundation | ✅ Complete | 3/3 |
| Phase 3: Tool Discovery | ✅ Complete | 4/4 |
| Phase 4: Documentation | ✅ Complete | 4/4 |
| Phase 5: Workflow Migration | 🔄 In Progress | 1/5 |
| Phase 5b: Protocol 128 Impl | ✅ Complete | 6/6 |
| Phase 6: Bash Script Migration | ❌ Deferred | 1/3 (Moved to Backlog) |
| Phase 7: Documentation | ✅ Complete | 4/4 |
| Phase N: Closure | ⏳ Pending | 0/5 |

**Total**: 28/41 tasks complete

