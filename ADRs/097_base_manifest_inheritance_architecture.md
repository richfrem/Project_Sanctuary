# ADR-097: Base Manifest Inheritance Architecture

## Status
Proposed

## Context
The project currently has multiple specialized manifest files for different bundling contexts (learning, guardian, bootstrap, red_team, learning_audit). Each manifest contains a `core` section with duplicated file lists, leading to:

1. **Duplication Risk**: Changes to core files (e.g., adding a new Protocol) must be updated in multiple manifests.
2. **Drift**: Manifests can become inconsistent over time.
3. **Complexity**: New bundle types require copying and maintaining the same core structure.

### Current Architecture
Each manifest explicitly defines its own `core` and `topic` sections:
```json
{
  "core": ["01_PROTOCOLS/128_Hardened_Learning_Loop.md", "ADRs/...", ...],
  "topic": ["specific_file.md"]
}
```

![Current Architecture](../docs/architecture_diagrams/workflows/bundling-architecture-current.png)

## Decision
Adopt a **Base Manifest Inheritance** pattern where:

1. **Base Manifests** are stored in `plugins/context-bundler/`
2. **Base Manifest Index** (`base-manifests-index.json`) maps type IDs to file paths
3. **Specialized Manifests** use an `extends` property to inherit from a base manifest
4. **Dynamic Topic Section** is managed via `manifest_manager.py` CLI (add/remove)

### Proposed Structure
```json
{
  "extends": "learning-core",
  "topic": ["ADRs/New_Research.md"]
}
```

### Resolution Flow
```
1. Workflow: workflow-bundle
2. manifest_manager.py init --type TYPE
   → Loads from base-manifests-index.json
   → Creates manifest with `extends` property
3. bundle.py manifest.json -o output.md
   → Resolves `extends` → loads base files
   → Merges with local `topic` section
4. LLM Activity (Research/Analysis)
   → manifest_manager.py add --path file.md --section topic
   → Rebundle → Recursive loop until complete
5. workflow-seal → workflow-persist
```

![Proposed Architecture](../docs/architecture_diagrams/workflows/bundling-architecture-proposed.png)

### Supported Base Types
| Type ID | Base Manifest | Purpose |
|:--------|:--------------|:--------|
| `generic` | `base-generic-file-manifest.json` | One-off bundles, no core |
| `learning-core` | `base-learning-core.json` | Protocol 128 learning seals |
| `learning-audit-core` | `base-learning-audit-core.json` | Red Team audit packets |
| `red-team-core` | `base-red-team-core.json` | Technical audit snapshots |
| `guardian-core` | `base-guardian-core.json` | Session bootloader context |
| `bootstrap-core` | `base-bootstrap-core.json` | Fresh repo onboarding |

## Consequences

### Positive
- **Single Source of Truth**: Core file lists are defined once in base manifests.
- **Automatic Propagation**: Updates to base manifests apply to all dependent bundles.
- **Simplified Maintenance**: Adding new bundle types requires only defining a base manifest.
- **Aligns with Protocol 128**: Supports recursive learning loops with stable foundation + evolving delta.

### Negative
- **Migration Effort**: Existing manifests must be refactored to use `extends` pattern.
- **Code Updates**: `bundle.py`, `operations.py`, and `cortex_cli.py` require modifications.
- **Index Maintenance**: `base-manifests-index.json` must be kept in sync.

## Alternatives Considered

### A: Shared Core Module (Refactor)
Extract bundling logic into `tools/shared/bundler_core.py` imported by both MCP and CLI.
- **Pros**: DRY for code logic.
- **Cons**: Doesn't address manifest duplication.

### B: Pipeline Approach (Atomic Tools)
Separate preparation (`manifest_compiler.py`) from execution (`bundle.py`).
- **Pros**: Unix philosophy.
- **Cons**: Multiple steps for user/agent.

### C: CLI Gateway
MCP shells out to `bundle.py` CLI.
- **Pros**: Extreme centralization.
- **Cons**: Slow, fragile for in-memory operations.

## Implementation
See Tasks: `specs/0002-spec-0002/tasks.md` (Phase 2: Migration)

## Related Documents
- [ADR 089: Modular Manifest Pattern](089_modular_manifest_pattern.md)
- [Protocol 128: Hardened Learning Loop](../01_PROTOCOLS/128_Hardened_Learning_Loop.md)
- [Protocol 130: Manifest Deduplication](../01_PROTOCOLS/130_Manifest_Deduplication_Protocol.md)
- [Design Proposal](../docs/architecture/designs/bundling-architecture-proposal.md)

## Diagrams
- Source (Proposed): [`bundling-architecture-proposed.mmd`](../docs/architecture_diagrams/workflows/bundling-architecture-proposed.mmd)
- Source (Current): [`bundling-architecture-current.mmd`](../docs/architecture_diagrams/workflows/bundling-architecture-current.mmd)
- Design Proposal: [`bundling-architecture-proposal.md`](../docs/architecture/designs/bundling-architecture-proposal.md)
