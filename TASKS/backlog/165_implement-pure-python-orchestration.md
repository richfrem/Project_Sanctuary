# Implement Pure Python Orchestration (ADR-096)

## Priority
High (Architectural Cleanup)

## Context
Per [ADR-096 (Pure Python Orchestration)](../../ADRs/096_pure_python_orchestration.md), we will migrate from the "Thick Python / Thin Shim" architecture (ADR-030 v2) to a fully Python-native model.

**Current State** (see [hybrid-spec-workflow.mmd](../../docs/diagrams/analysis/sdd-workflow-comparison/hybrid-spec-workflow.mmd)):
- User invokes `/sanctuary-start` → Thin Shim (`workflow-start.sh`) → Python CLI → `WorkflowManager`
- The Thin Shim layer (`scripts/bash/*.sh`) is now pure passthrough (`exec python3 tools/cli.py ...`)

**Target State**:
- User invokes `/sanctuary-start` → Python CLI → `WorkflowManager` (no shim)
- Single truth: Workflow Markdown → `python tools/cli.py workflow <command>`

## Core Asset
The [WorkflowManager](../../tools/orchestrator/workflow_manager.py) class already handles:
- `start_workflow()` — Git checks, branch creation, spec initialization
- `run_retrospective()` — Proof check, interactive feedback
- `end_workflow()` — Commit and push
- `cleanup_workflow()` — Post-merge branch cleanup

This task removes the `.sh` shim layer and updates workflow markdowns to invoke `cli.py` directly.

## Migration Examples

### Workflow Markdown Invocation (Before → After)

| Current (Shim) | Target (Pure Python) |
|----------------|---------------------|
| `source scripts/bash/sanctuary-start.sh` | `python tools/cli.py workflow start` |
| `source scripts/bash/adr-manage.sh` | `python tools/cli.py workflow start --name codify-adr` |
| `source scripts/bash/spec-kitty.implement.sh` | `python tools/cli.py workflow start --name spec-kitty.implement` |

### Full Command Mapping

```
# Start a workflow
OLD: source scripts/bash/sanctuary-start.sh "$SPEC_ID"
NEW: python tools/cli.py workflow start --target "$SPEC_ID"

# End workflow (commit + push)
OLD: source scripts/bash/sanctuary-end.sh
NEW: python tools/cli.py workflow end

# Post-merge cleanup
OLD: source scripts/bash/workflow-cleanup.sh
NEW: python tools/cli.py workflow cleanup

# Retrospective
OLD: source scripts/bash/sanctuary-retrospective.sh
NEW: python tools/cli.py workflow retrospective
```

### Call Chain (Simplified)

```
BEFORE:
  User → Workflow Markdown → .sh shim → cli.py → WorkflowManager → Git/Files

AFTER:
  User → Workflow Markdown → cli.py → WorkflowManager → Git/Files
```

## Requirements

### Phase 1: Delete Shims
- [ ] Remove `scripts/bash/sanctuary-start.sh`
- [ ] Remove `scripts/bash/codify-*.sh` (all codify shims)
- [ ] Audit `scripts/bash/` for any other workflow-related shims

### Phase 2: Update Workflow Markdowns (~27 files)
- [ ] Find & Replace in all `.agent/workflows/*.md` files:
    - **Old**: `source scripts/bash/sanctuary-start.sh ...`
    - **New**: `python tools/cli.py workflow start ...`
- [ ] Update any `// turbo` annotations if needed

### Phase 2.5: Update Documentation Referencing Shims
- [ ] **Create temp migration script** (`scripts/migrate_shims_to_cli.py`):
    - Scan `scripts/bash/*.sh` to build mapping of `.sh` → `cli.py` commands
    - Find/replace across: `docs/`, `ADRs/`, `LEARNING/`, `protocols/`, `.agent/`, `README*.md`
    - Output report of changes made
    - Delete script after migration complete
- [ ] Grep for `.sh` references across docs: `grep -r "scripts/bash" docs/ ADRs/ LEARNING/ protocols/ .agent/`
- [ ] Update ADRs that mention shim invocation (e.g., ADR-030, ADR-036)
- [ ] Update protocol docs (Protocol 128 references)
- [ ] Update any README files with workflow examples
- [ ] Update `.agent/rules/` if workflow invocation patterns are documented there

### Phase 3: Create New Diagram
- [ ] Create `pure-python-workflow.mmd` (evolution of `hybrid-spec-workflow.mmd` without Thin Shim)
- [ ] Render PNG
- [ ] Keep existing `hybrid-spec-workflow.mmd` as historical reference (or mark deprecated)

### Phase 4: Verification
- [ ] Test `python tools/cli.py workflow start` directly
- [ ] Test `python tools/cli.py workflow end`
- [ ] Test `python tools/cli.py workflow cleanup`
- [ ] Run through a full spec lifecycle without shims

### Phase 5: Create Shareable Migration Guide (Output Artifact)
- [ ] Create `LEARNING/topics/pure_python_orchestration/agent_migration_guide.md`:
    - **Context**: Why eliminate shell shims (fragility, platform issues, triple-tracking overhead)
    - **Prerequisites**: What the target project needs (Python CLI, orchestrator class)
    - **Step-by-step tasks** an LLM agent can follow:
        1. Inventory existing `.sh` shim scripts
        2. Create command mapping table (`.sh` → Python CLI)
        3. Build migration script to find/replace references
        4. Delete shims
        5. Update workflow/documentation files
        6. Create/update architecture diagram
        7. Verification checklist
    - **Example code snippets** from this migration
    - **Common pitfalls** and how to avoid them
- [ ] Include a template `cli.py workflow` command structure they can adapt
- [ ] Link to this task as the reference implementation

## References
- [ADR-096: Pure Python Orchestration](../../ADRs/096_pure_python_orchestration.md)
- [ADR-030: Workflow Architecture (v2)](../../ADRs/030_workflow_architecture.md)
- [WorkflowManager](../../tools/orchestrator/workflow_manager.py)
- [Hybrid Spec Workflow Diagram](../../docs/diagrams/analysis/sdd-workflow-comparison/hybrid-spec-workflow.mmd)