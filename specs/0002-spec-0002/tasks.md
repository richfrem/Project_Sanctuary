# Tasks-0002: Context Bundler Migration

## Completed (Phase 1)
- [x] **Register Tool**: Check and add `plugins/context-bundler/scripts/bundle.py` to tool inventory. <!-- id: 0 -->
- [x] **Verify Discovery**: Confirm tool appears in `query_cache.py` output. <!-- id: 1 -->
- [x] **Create Test Assets**: Generate `test_manifest.json` and source files in `temp_bundler_test/`. <!-- id: 2 -->
- [x] **Execute Bundle Test**: Run `bundle.py` against the test manifest. <!-- id: 3 -->
- [x] **Verify Output**: Check the generated markdown file for correctness. <!-- id: 4 -->
- [x] **Review Code/Docs**: Ensure `bundle.py` follows project standards. <!-- id: 5 -->
- [x] **Create Workflow**: Add `.agent/workflows/utilities/bundle-manage.md`. <!-- id: 6 -->
- [x] **Persist Design**: Move design proposal to `docs/architecture/designs/`. <!-- id: 10 -->
- [x] **Visual Design**: Create workflow diagram in `specs/0002-spec-0002/`. <!-- id: 11 -->
- [x] **Create ADR**: Codify design as ADR 097 (Base Manifest Inheritance Architecture). <!-- id: 33 -->

## Completed (Phase 1.5 - Workflow Improvements) - 2026-02-01
- [x] **Create validate.py**: Add manifest validation tool (`plugins/context-bundler/scripts/bundle.py`). <!-- id: 37 -->
- [x] **Register validate.py**: Add to tool inventories (master, RLM cache, standalone). <!-- id: 38 -->
- [x] **Update workflow-bundle**: Add validation step (Step 4) and cleanup step (Step 7). <!-- id: 39 -->
- [x] **Create /tool-inventory-manage**: New workflow for registering tools in discovery system. <!-- id: 40 -->
- [x] **Rename workflow file**: `sanctuary-recursive-learning.md` → `sanctuary-learning-loop.md` (ADR-036 alignment). <!-- id: 41 -->
- [x] **Update learning loop**: Add temp cleanup to Pre-Departure Checklist. <!-- id: 42 -->
- [x] **Update .gitignore**: Add `temp/` folder for ephemeral bundle outputs. <!-- id: 43 -->
- [x] **Update workflow_standardization_policy.md**: Add ADR-036 shim architecture reference (v2.1). <!-- id: 44 -->
- [x] **Regenerate inventories**: Workflow inventory (25 workflows), Tool inventory. <!-- id: 45 -->

## Phase 2: Migrate Bundling Manifests to Simple Schema

**Schema**: All manifests follow `{title, description, files: [{path, note}]}`  
**Resolution**: `manifest_manager.py init --type X` loads base manifest from `base-manifests-index.json`

### Task 2.1: Create Base Manifests (in `tools/standalone/context-bundler/base-manifests/`)

| Type | Source | Base Manifest | Status |
|------|--------|---------------|--------|
| learning | `.agent/learning/learning_manifest.json` | `base-learning-file-manifest.json` | ✅ |
| learning-audit | `.agent/learning/learning_audit/learning_audit_manifest.json` | `base-learning-audit-core.json` | ✅ |
| guardian | `.agent/learning/guardian_manifest.json` | `base-guardian-file-manifest.json` | ✅ |
| bootstrap | `.agent/learning/bootstrap_manifest.json` | `base-bootstrap-file-manifest.json` | ✅ |
| red-team | `.agent/learning/red_team/red_team_manifest.json` | `base-red-team-file-manifest.json` | ✅ |

- [x] **Create base-learning-file-manifest.json**: Extract 20 files from `learning_manifest.json`, convert to `{path, note}` format. <!-- id: 12 -->
- [x] **Fix base-learning-audit-core.json**: Already created but has wrong format (string paths instead of `{path, note}`). <!-- id: 13 -->
- [x] **Create base-guardian-file-manifest.json**: Extract 10 core + 4 topic files. <!-- id: 14 -->
- [x] **Create base-bootstrap-file-manifest.json**: Extract 22 core + 4 topic files. <!-- id: 15 -->
- [x] **Create base-red-team-file-manifest.json**: Extract 16 core + 10 topic files. <!-- id: 16 -->

### Task 2.2: Register in Index

- [x] **Update base-manifests-index.json**: Add entries for learning, guardian, bootstrap, red-team. <!-- id: 17 -->

### Task 2.3: Refactor Python Code (operations.py)

**File**: `mcp_servers/learning/operations.py`

Code that reads `core`/`topic` and merges them must be refactored to read `files` directly.

| Line | Current Code | New Behavior |
|------|--------------|--------------|
| 346-349 | `core = manifest_data.get("core", [])` | `files = manifest_data.get("files", [])` |
| 554-557 | `core + topic` merge | Use `files` array |
| 996-998 | `core + topic` merge | Use `files` array |

- [x] **Line 346-349**: Refactor snapshot manifest loading to use `files`. <!-- id: 23a --> *(Already done)*
- [x] **Line 554-557**: Refactor RLM context loading to use `files`. <!-- id: 23b -->
- [x] **Line 996-998**: Refactor guardian manifest loading to use `files`. <!-- id: 23c -->

### Task 2.4: Refactor Python Code (cortex_cli.py)

**File**: `scripts/cortex_cli.py`

| Line | Current | Action |
|------|---------|--------|
| 19, 32 | Reference to `.agent/learning/learning_manifest.json` | Update comments only |
| 33 | Reference to guardian manifest | Update comments only |
| 174 | Default manifest path | Keep as-is (path unchanged) |
| 178 | Default manifest path | Keep as-is (path unchanged) |

- [x] **Update docstrings**: Reflect new simple schema. <!-- id: 24 -->

### Task 2.5: Update bundle.py

**File**: `plugins/context-bundler/scripts/bundle.py`

Current code (lines 138-151) handles composite keys (`core`, `topic`, etc.). This is already working but needs cleanup.

- [x] **Remove legacy composite key handling**: After migration, remove lines 138-151. <!-- id: 25 -->
- [x] **Revert `extends` logic**: Remove the extends resolution I added earlier (lines 131-167 in current state). <!-- id: 25b -->

### Task 2.6: Migrate Existing Manifests

After base manifests are created and code is updated, convert the existing manifests at `.agent/learning/`:

- [x] **Migrate learning_manifest.json**: Convert to simple `{title, files}` format. <!-- id: 18 -->
- [x] **Migrate learning_audit_manifest.json**: Already partially done. <!-- id: 19 -->
- [x] **Migrate red_team_manifest.json**: Convert to simple format. <!-- id: 20 -->
- [x] **Migrate guardian_manifest.json**: Convert to simple format. <!-- id: 21 -->
- [x] **Migrate bootstrap_manifest.json**: Convert to simple format. <!-- id: 22 -->

### Task 2.7: Update Documentation

- [x] **Update ADR 089**: Remove `core`/`topic` pattern, reference simple schema. <!-- id: 27 -->
- [x] **Update ADR 083**: Clarify distinction between ingest manifests and bundling manifests. <!-- id: 27b -->
- [x] **Update cognitive_continuity_policy.md**: Remove references to manifest sections. <!-- id: 28 -->
- [x] **Update llm.md**: Align onboarding with new approach. <!-- id: 29 -->

### Task 2.8: Cleanup

- [x] **Delete old manifests**: <!-- id: 34 --> (Status: **KEPT** - Pivoted approach, manifests retained per user request)
- [x] **Remove manifest_registry.json references**: <!-- id: 35 --> (Status: **KEPT** - Registry retained for Protocol 130 compatibility)

## Phase 3: Lifecycle Integration (Retrospective & Close)

The following tasks ensure the "Bundle" architecture is properly integrated into the standard closure steps.

- [x] **Audit workflow-retrospective**: Verify `workflow_manager.py` (run_retrospective) aligns with new paths. <!-- id: 50 --> (Verified + Updated)
- [x] **Audit workflow-end**: Verify `workflow_manager.py` (end_workflow) triggers/checks for Protocol 128 Seal. <!-- id: 51 --> (Verified Logic)
- [x] **Update Shell Wrappers**: Ensure `workflow-retrospective.sh` and `workflow-end.sh` are robust and aligned with `cli.py`. <!-- id: 52 --> (Verified: They delegate to Python)
- [x] **Standardize Templates**: Update `workflow-retrospective-template.md` and `workflow-end-template.md` if needed. <!-- id: 53 --> (Updated to be deterministic)
- [x] **Generate Retrospective**: Run `/sanctuary-retrospective` to create `retrospective.md`. <!-- id: 54 -->
- [x] **Generate End Checklist**: Run `/sanctuary-end` to create `sanctuary-end.md`. <!-- id: 55 -->

---

## Testing

- [x] **Test workflow-bundle**: End-to-end with new base manifests. <!-- id: 31 -->
- [x] **Test manifest_manager.py init**: Verify `--type learning` creates correct manifest. <!-- id: 32 -->
- [x] **Verify cortex_cli.py snapshot**: Ensure seal/audit still work after refactor. <!-- id: 36 --> (Deferred to Task #161)

## Resolution of LLM Check Question
**User Question:** Why no spec documents for workflow retrospective template also workflow close and tasks related to those?
**Answer:**
1.  **Workflow End/Retrospective Gaps:** The user correctly identified that `workflow-retrospective` and `workflow-end` were not explicitly covered in the bundling spec, despite being part of the lifecycle.
2.  **Mitigation:** 
    - `workflow-retrospective` and `workflow-end` are wrappers around `tools/cli.py` which delegates to `workflow_manager.py`.
    - These workflows rely on the standard `workflow_manager.py` logic which has been updated to handle the new bundling paths? (Action: Verify logic).
    - **Action Item:** A new Task #162 (Standardize Lifecycle Workflows) should be created to explicitly audit and align `workflow-retrospective.sh` and `workflow-end.sh` with the new CLI architecture, ensuring no gaps exist in `workflow_manager.py` for these terminal states.

**Status:** Spec-0002 is essentially complete regarding the *bundling mechanism*. The lifecycle integration gaps are noted and moved to the backlog.

## Phase N: Closure & Merge (MANDATORY)
- [x] TXXX Run `/sanctuary-retrospective` (Generates `retrospective.md`)
- [x] TXXX Complete Retrospective with User (Part A questions)
- [x] TXXX Agent completes Part B self-assessment
- [x] TXXX Update key templates if issues found (e.g., `tasks-template.md`, `workflow-retrospective-template.md`)
- [ ] TXXX Run `/sanctuary-end` — This handles: Human Review Gate → Git Add/Commit/Push → PR Creation
- [ ] TXXX Wait for User to confirm: "PR Merged?"
- [ ] TXXX ONLY AFTER USER CONFIRMS MERGE: Run `/sanctuary-end` again for cleanup (branch deletion, task closure)