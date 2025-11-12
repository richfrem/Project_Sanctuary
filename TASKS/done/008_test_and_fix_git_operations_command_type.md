```markdown
# 008 Test & Fix Git Command Types (git_operations)

Status: Completed

Completed: 2025-11-11

## Completion Summary

The git operations command type has been significantly enhanced beyond the original scope:

### ✅ Issues Resolved
- **KeyError on missing `output_artifact_path`**: Fixed by ensuring all command schemas require this field
- **Deleted file handling**: Implemented proper `files_to_remove` feature with explicit `git rm` staging
- **Manifest generation**: Fixed empty manifest issues for deletion-only commits
- **Protocol 101 compliance**: Maintained integrity while adding deletion support

### 🚀 Major Enhancements Implemented
1. **New `files_to_remove` field**: Added to command schema for explicit deletion handling
2. **Phase 1.5 git rm logic**: Dedicated deletion staging before additions
3. **Updated documentation**: `command_schema.md` and `howto-commit-command.md` reflect new features
4. **Clean architecture**: Separated addition and deletion logic for maintainability

### 📋 Final Status
- All git operation bugs fixed
- New deletion feature fully implemented and tested
- Documentation updated
- Command schema extended
- Ready for production use

The orchestrator now handles git operations robustly with explicit support for both file additions and deletions.

Purpose

- Create tests and a small fix to ensure `git_operations` (MECHANICAL_GIT) command payloads are validated and executed consistently by the orchestrator.
- Resolve validator vs executor mismatches (e.g., missing `output_artifact_path` causing runtime KeyError).

Background

- The orchestrator detects command*.json files and routes them through `orchestrator/commands.py` for parsing/validation and `orchestrator/gitops.py` for mechanical git execution.
- Currently the validator accepts objects with a top-level `git_operations` key as a MECHANICAL_GIT command, but runtime code paths elsewhere may assume `output_artifact_path` is present and access it unguarded. This causes failures when command JSON lacks that field.
- A previous run produced a KeyError: `'output_artifact_path'` when processing a git command file.

Acceptance criteria

- Unit tests covering the following scenarios are added (pytest style):
  - Valid `git_operations` command is accepted by `orchestrator/commands.py` and executed by `gitops.execute_mechanical_git` without raising KeyError.
  - Missing `files_to_add` or `commit_message` in `git_operations` should be rejected with a clear validation error.
  - A command without `output_artifact_path` should either be accepted if the executor can handle its absence or rejected by the validator — behavior must be explicit and consistent.
  - Paths in `files_to_add` that don't exist are skipped with a warning and the manifest generated contains only existing files.
  - `push_to_origin: false` runs add+commit locally (or creates commit_success=False if nothing to commit) without attempting network push.

- The validator and executor agree on required/optional fields for MECHANICAL_GIT.

- The orchestrator no longer throws KeyError for missing `output_artifact_path`; instead it either writes to a default artifact path or validator forces the caller to provide it.

- A short README section (in `council_orchestrator/README.md` or `TASKS/backlog/008_*`) documents the expected command JSON schema for `git_operations`.

Implementation notes / proposed approach

1. Reproduce (unit test): Create a small test harness that invokes `parse_command_from_json` with representative JSON and asserts validator result.
2. Add executor guard: In `orchestrator/app.py` (or the main task runner), guard accesses to `output_artifact_path` and fall back to a default like `council_orchestrator/command_results/<timestamp>_gitop.json` when safe.
3. Alternatively (preferred): Tighten the validator in `orchestrator/commands.py` so MECHANICAL_GIT requires `output_artifact_path`. This keeps behavior explicit and surfaces missing fields earlier.
4. Add pytest cases under `council_orchestrator/tests/test_commands_gitops.py` verifying both validation and execution flows (mock filesystem or use tmp_path fixtures to create files).
5. Run the tests and ensure orchestrator processes a `command_git_ops.json` sample successfully in dry-run mode (push_to_origin:false).
6. Update TASKS/backlog and PR notes describing the change.

Notes

- Be careful with the `commit_manifest.json` generation: `gitops.execute_mechanical_git` writes that file to the repo root; tests should clean up or use temporary directories.
- Keep `push_to_origin` default to false in tests to avoid network operations.

References

- `council_orchestrator/orchestrator/commands.py`
- `council_orchestrator/orchestrator/gitops.py`
- Orchestrator logs and previous KeyError stack traces

Estimated effort

- 2-4 hours: tests + small validation change + executor guard + test run

Owner

- Assigned to: team (or self).
```
