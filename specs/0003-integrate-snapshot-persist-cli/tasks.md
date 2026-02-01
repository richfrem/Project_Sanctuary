---
description: "Tasks for Integrate Snapshot and Persist-Soul into CLI"
---

# Tasks: Integrate Snapshot and Persist-Soul into CLI

**Input**: Design documents from `/specs/0003-integrate-snapshot-persist-cli/`
**Prerequisites**: plan.md

## Phase 0: Pre-Flight
- [x] T000 Check for existing branch `spec/0003-integrate-snapshot-persist-cli`
- [x] T001 If clean, create/checkout branch `spec/0003-integrate-snapshot-persist-cli`
- [x] T002 Initialize Spec Bundle (`spec.md`, `plan.md`) via `/workflow-start` (Done)

## Phase 1: Setup & Analysis
- [x] T003 Analyze `mcp_servers/learning/operations.py` to understand direct usage requirements.
- [x] T004 Run `tools/codify/rlm/distiller.py` via `cli.py` (if possible) or directly to register `tools/cli.py` and related tools in `tools/tool_inventory.json` and `rlm_tool_cache.json`.
- [x] T004b Register additional Hugging Face support tools (`upload_to_huggingface.py`, `hf_utils.py`, etc.) ensuring standardized headers and RLM cache entries.

## Phase 2: Implementation (User Story 1 & 2)
- [x] T005 Refactor `tools/cli.py` import structure to ensure `mcp_servers` is accessible.
- [x] T006 Update `tools/cli.py`: Rewrite `snapshot` command to use `LearningOperations.capture_snapshot`.
- [x] T007 Update `tools/cli.py`: Add `persist-soul`, `debrief`, and `guardian` commands using `LearningOperations`.
- [x] T007b Verify `cli.py` commands (`snapshot`, `persist-soul`, `debrief`, `guardian`) structure.

## Phase 3: Verification
- [x] T008 Verify `snapshot` command with `learning_audit` type.
- [x] T009 Verify `debrief` function.

## Phase 4: Closure
- [ ] T010 Run `/workflow-retrospective`
- [ ] T011 Run `/workflow-end`
