# Task: Scan for Stale File References After Docs Restructuring

**Created:** 2026-01-02
**Priority:** Medium
**Status:** Pending

## Context

A major documentation restructuring was performed on 2026-01-02 that moved files from the project root to `docs/` subdirectories. This task ensures no stale references remain.

## Files Moved

| Original Location | New Location |
|-------------------|--------------|
| `chrysalis_core_essence.md` | `docs/philosophy/` |
| `The_Garden_and_The_Cage.md` | `docs/philosophy/` |
| `Council_Inquiry_Gardener_Architecture.md` | `docs/architecture/` |
| `PROJECT_SANCTUARY_SYNTHESIS.md` | `docs/` |
| `Socratic_Key_User_Guide.md` | `docs/tutorials/` |
| `GARDENER_TRANSITION_GUIDE.md` | `docs/tutorials/` |
| `ENVIRONMENT.md` | `docs/operations/` |
| `DEPENDENCY_MANIFEST.md` | `docs/operations/` |

## Already Updated

- [x] `mcp_servers/lib/ingest_manifest.json`
- [x] `tasks/backlog/023_dependency_management_and_environment_reproducibility.md`
- [x] `docs/INDEX.md`

## Scan Commands

Run these to find any remaining stale references:

```bash
# Find references to moved files
grep -rl "chrysalis_core_essence.md" . --include="*.md" --include="*.json" | grep -v node_modules
grep -rl "The_Garden_and_The_Cage.md" . --include="*.md" --include="*.json" | grep -v node_modules
grep -rl "Council_Inquiry_Gardener_Architecture.md" . --include="*.md" --include="*.json" | grep -v node_modules
grep -rl "PROJECT_SANCTUARY_SYNTHESIS.md" . --include="*.md" --include="*.json" | grep -v node_modules
grep -rl "Socratic_Key_User_Guide.md" . --include="*.md" --include="*.json" | grep -v node_modules
grep -rl "GARDENER_TRANSITION_GUIDE.md" . --include="*.md" --include="*.json" | grep -v node_modules
grep -rl "ENVIRONMENT.md" . --include="*.md" --include="*.json" | grep -v node_modules
grep -rl "DEPENDENCY_MANIFEST.md" . --include="*.md" --include="*.json" | grep -v node_modules
```

## Acceptance Criteria

- [ ] Run all scan commands
- [ ] Update any stale references found
- [ ] Verify `docs/INDEX.md` links work
- [ ] Test that RAG ingestion still works with new paths
