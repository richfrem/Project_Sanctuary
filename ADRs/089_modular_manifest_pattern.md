# ADR 089: Modular Manifest Pattern for Context-Aware Snapshots

**Status:** PROPOSED  
**Date:** 2026-01-03  
**Author:** Antigravity (Agent) / User (Steward)

---

## Context

Protocol 128 (Cognitive Continuity) requires generating context packets for various use cases:
- **Session Handover**: Sealing knowledge for successor agents
- **Learning Audits**: Red Team review of research artifacts
- **Guardian Wakeup**: Protocol 128 bootloader context
- **Onboarding**: Fresh repository setup for new developers

Previously, snapshot generation was tightly coupled to a single manifest (`learning_manifest.json`). As the project evolved, different contexts required different file subsets, leading to either:
1. A bloated single manifest that tried to serve all purposes
2. Ad-hoc file lists passed manually to tools

This created maintenance burden and inconsistent context quality across use cases.

## Decision

Establish a **Modular Manifest Pattern** where:

### 1. Manifest per Use Case
Each distinct context has its own manifest file in `.agent/learning/`:

| Manifest | Purpose | Primary Files |
|----------|---------|---------------|
| `learning_manifest.json` | Session seal (successor context) | ADRs, Protocols, Learning artifacts |
| `learning_audit_manifest.json` | Red Team review | Research topics, sources, prompts |
| `guardian_manifest.json` | Protocol 128 bootloader | Identity anchor, primer, calibration |
| `bootstrap_manifest.json` | Fresh repo onboarding | BOOTSTRAP.md, Makefile, ADR 073, requirements |
| `red_team_manifest.json` | Technical audit | Git diff targets, modified files |

### 2. Shared Mechanics via CLI
The `scripts/cortex_cli.py` provides unified commands that accept manifest paths:

```bash
# Guardian wakeup with specific manifest
python3 scripts/cortex_cli.py guardian --manifest .agent/learning/guardian_manifest.json

# Bootstrap context for onboarding
python3 scripts/cortex_cli.py bootstrap-debrief --manifest .agent/learning/bootstrap_manifest.json

# Snapshots with type-specific manifests
python3 scripts/cortex_cli.py snapshot --type seal --manifest .agent/learning/learning_manifest.json
python3 scripts/cortex_cli.py snapshot --type learning_audit --manifest .agent/learning/learning_audit/learning_audit_manifest.json
```

### 3. Manifest Schema
All manifests follow a simple JSON array of relative paths:

```json
[
    "README.md",
    "docs/operations/BOOTSTRAP.md",
    "Makefile",
    "ADRs/073_standardization_of_python_dependency_management_across_environments.md"
]
```

- **Directories** can be included (e.g., `"LEARNING/topics/my_topic/"`) for recursive capture
- **Relative paths** from project root
- **No metadata** in manifest—content processing extracts structure

### 4. Evolution Path
New use cases are added by:
1. Creating a new manifest file (`.agent/learning/<use_case>_manifest.json`)
2. Optionally adding a dedicated CLI command to `cortex_cli.py`
3. Documenting the use case in this ADR or a dedicated workflow

## Consequences

### Positive
- **Separation of Concerns**: Each manifest is optimized for its specific context
- **Evolvable**: Manifests can be updated independently as requirements change
- **Reusable Mechanics**: Snapshot generation logic is shared across all use cases
- **Auditable**: Each manifest explicitly declares its scope
- **Onboarding**: New developers/agents get targeted context, not full genome

### Negative
- **Manifest Proliferation**: Risk of too many manifests if not curated
- **Coordination**: Changes to core files may require updating multiple manifests
- **Content Duplication**: Generated outputs (e.g., `learning_package_snapshot.md`) may contain content from source manifests; including both wastes tokens

### Mitigation
- **Manifest Registry**: Maintain this ADR as the canonical list of active manifests
- **Gardener Checks**: Include manifest health in TDA gardener scans
- **Deduplication Strategy**: Mark generated files with `[GENERATED_FROM: manifest_name]` metadata; snapshot tool should detect and exclude files that are outputs of included manifests

### Implemented: Manifest Deduplication (Protocol 130)

> [!NOTE]
> **Status:** Implemented (Jan 2026) via `manifest_registry.json` and `operations.py`.
>
> **Protocol 130 Strategy:**
> 1. Maintains a **Manifest Registry** (`.agent/learning/manifest_registry.json`) mapping manifests to their output files.
> 2. During snapshot generation, the system checks if any included files are themselves generated outputs of other included manifests.
> 3. Automatically removes the source manifests if their output is already included, preventing token waste.
> 4. See [Protocol 130](../01_PROTOCOLS/130_Manifest_Deduplication_Protocol.md) for full details.


## Manifest Inventory (Current)

### Protocol 128 Learning Manifests
Manifests used by `cortex_cli.py` for cognitive continuity workflows:

| Manifest | Path | CLI Buildable | Command |
|----------|------|:-------------:|---------|
| Learning Seal | `.agent/learning/learning_manifest.json` | ✅ | `snapshot --type seal` |
| Learning Audit | `.agent/learning/learning_audit/learning_audit_manifest.json` | ✅ | `snapshot --type learning_audit` |
| Red Team Audit | `.agent/learning/red_team/red_team_manifest.json` | ✅ | `snapshot --type audit` |
| Guardian | `.agent/learning/guardian_manifest.json` | ✅ | `guardian` |
| Bootstrap | `.agent/learning/bootstrap_manifest.json` | ✅ | `bootstrap-debrief` |

**Usage Examples:**
```bash
# Session seal for successor agent
python3 scripts/cortex_cli.py snapshot --type seal --manifest .agent/learning/learning_manifest.json

# Learning audit for Red Team review
python3 scripts/cortex_cli.py snapshot --type learning_audit --manifest .agent/learning/learning_audit/learning_audit_manifest.json

# Technical audit with git diff verification
python3 scripts/cortex_cli.py snapshot --type audit --manifest .agent/learning/red_team/red_team_manifest.json

# Guardian wakeup (Protocol 128 bootloader)
python3 scripts/cortex_cli.py guardian --manifest .agent/learning/guardian_manifest.json

# Bootstrap debrief for fresh repo onboarding
python3 scripts/cortex_cli.py bootstrap-debrief --manifest .agent/learning/bootstrap_manifest.json
```

### System Manifests
Manifests used by MCP servers and infrastructure (not directly CLI-buildable):

| Manifest | Path | CLI Buildable | Purpose |
|----------|------|:-------------:|---------|
| Exclusion | `mcp_servers/lib/exclusion_manifest.json` | ❌ | Files/patterns to exclude from ingestion |
| Ingest | `mcp_servers/lib/ingest_manifest.json` | ❌ | Files/directories to include in RAG ingestion |
| Registry | `.agent/learning/manifest_registry.json` | ❌ | Map of manifests to outputs (Protocol 130) |

### Forge & Dataset Manifests
Manifests for model training and HuggingFace (not directly CLI-buildable):

| Manifest | Path | CLI Buildable | Purpose |
|----------|------|:-------------:|---------|
| Forge (Model) | `forge/gguf_model_manifest.json` | ❌ | GGUF model metadata (architecture, params) |
| Ingest | `mcp_servers/lib/ingest_manifest.json` | ❌ | Dataset content sources (used by Forge/RAG) |
| HF Dataset | `hugging_face_dataset_repo/metadata/manifest.json` | ❌ | Soul persistence dataset metadata |
| Root | `manifest.json` | ❌ | Project-level manifest (snapshot generation) |

> [!NOTE]
> **Structural Variance:** Unlike Protocol 128 manifests (simple file arrays), the Forge manifest uses GGUF-specific schema with model architecture, parameters, and template fields. The Ingest manifest (`mcp_servers/lib/ingest_manifest.json`) follows a categorized directory structure for RAG and Forge content sources.

### Legacy/Deprecated

| Manifest | Path | Status |
|----------|------|--------|
| `manifest_learning_audit.json` | `.agent/learning/learning_audit/` | ❌ Removed (Jan 2026) |
| `manifest_seal.json` | `.agent/learning/` | ❌ Removed (Jan 2026) |

> [!TIP]
> When creating a new manifest, follow the naming convention `<use_case>_manifest.json` and place it in the appropriate domain directory.

---

## LLM Entry Point: `llm.md`

### Industry Context

An emerging standard called **`llm.txt`** (similar to `robots.txt` for web crawlers) is being adopted by companies like **Modular, ReadMe, and Prepr** to provide AI coding assistants with structured context.

> "By improving the clarity, structure, and directness of READMEs for human readability, you inherently enhance their utility for AI agents." — [benhouston3d.com](https://benhouston3d.com)

### Sanctuary Implementation

Project Sanctuary adopts `llm.md` as the standard entry point for AI coding assistants (Antigravity, Copilot, Claude Code, Cursor).

**Design Principles:**
1. **Pointer, Not Duplicate:** `llm.md` references `bootstrap_packet.md` rather than duplicating content
2. **Leverages Manifests:** Uses the bootstrap manifest (this ADR) for packet generation
3. **Token-Conscious:** Provides stats (~44K tokens) for context window planning
4. **Regenerable:** CLI command ensures freshness

**File Location:** `/llm.md` (repository root)

**Usage:**
```bash
# Regenerate the bootstrap packet
python3 scripts/cortex_cli.py bootstrap-debrief

# AI reads llm.md → finds bootstrap_packet.md → ingests context
```

### References

- [llm.txt Standard](https://llmstxt.org) — Emerging industry pattern
- [Modular's Implementation](https://docs.modular.com/llms.txt)
- [Model Context Protocol (MCP)](https://modelcontextprotocol.io) — Anthropic's context standard

---

## Related Documents

- [ADR 071: Protocol 128 (Cognitive Continuity)](./071_protocol_128_cognitive_continuity.md)
- [ADR 083: Manifest-Centric Architecture](./083_manifest_centric_architecture.md)
- [Protocol 128: Hardened Learning Loop](../01_PROTOCOLS/128_Hardened_Learning_Loop.md)
