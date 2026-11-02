# ADR 083: Manifest-Centric Architecture (The Single Source of Truth)

**Status**: Accepted
**Date**: 2025-12-28
**Context**: Protocol 128 (Harmonization)

## Context
Previously, Project Sanctuary's various subsystems (RAG Ingestion, Forge Fine-Tuning, Code Snapshots, and Soul Persistence) used disparate methods for defining their "scope":
-   **RAG**: Hardcoded list of directories in `operations.py`.
-   **Forge**: Custom regex and file walking in `forge_whole_genome_dataset.py`.
-   **Snapshots**: Ad-hoc `os.walk` or manual file lists in `snapshot_utils.py`.
-   **Exclusions**: Scattered across `exclusion_config.py` and local variable lists.

This led to a "split brain" problem where the Agent's RAG memory might know about file X, but its Fine-Tuning dataset (Forge) missed it, and the Audit Snapshot (Red Team) saw something else entirely. Exclusion rules were also applied inconsistently, leading to `node_modules` or `__pycache__` leaking into datasets.

## Decision
We are shifting to a **Manifest-Centric Architecture**. 
Two JSON files now serve as the Single Source of Truth (SSOT) for the entire system:

1.  **`mcp_servers/lib/ingest_manifest.json` (The "Include" List)**:
    -   Defines the **Base Genome**: The core set of files and directories that constitute the agent's identity and knowledge.
    -   Defines **Target Scopes**: Specific subsets for RAG (`unique_rag_content`), Forge (`unique_forge_content`), and Soul (`unique_soul_content`).
    -   **Rule**: If it's not in the manifest, it doesn't exist to the Agent's higher functions.

2.  **`mcp_servers/lib/exclusion_manifest.json` (The "Exclude" List)**:
    -   Defines universal blocking rules (`exclude_dir_names`, `always_exclude_files`, `exclude_patterns`).
    -   **Rule**: These rules are applied *after* inclusion, acting as a final firewall. `ContentProcessor` enforces this globally.

## Implementation Details

### 1. Unified Content Processor
A shared library (`mcp_servers/lib/content_processor.py`) drives all content access.
-   **Input**: A Manifest Scope (e.g., `common_content` + `rag_targets`).
-   **Process**: 
    1.  Traverses targets.
    2.  Apply `exclusion_manifest` logic (Protocol 128).
    3.  Parses/Validates Syntax (AST-based for Python).
    4.  Transforms to destination format (Markdown for RAG, JSONL for Forge).
-   **Output**: Clean, validated, harmonized data.

### 2. Subsystem Updates
-   **RAG Cortex**: Now iterates the manifest instead of walking the filesystem blindly.
-   **Architecture Forge**: Generates datasets strictly from the manifest, ensuring the fine-tuned model matches the RAG knowledge base.
-   **Snapshots (CLI)**: Default behavior now snapshots the "Base Genome" from the manifest, ensuring audits match reality.

## Consequences
### Positive
-   **Consistency**: "What you see is what you get" across all agent modalities.
-   **Security**: Single point of control for exclusions (preventing secret leakage).
-   **Maintainability**: Adding a new directory to the Agent's scope is a one-line JSON change, not a code refactor.
-   **Integrity**: Syntax errors in source code are caught during ingestion (by `ContentProcessor`), preventing garbage data in RAG/Forge.

### Negative
-   **Rigidity**: "Quick tests" outside the manifest require updating the JSON or using specific override flags.
-   **Dependency**: All tools now strictly depend on `content_processor.py` and the JSON manifests.

## Compliance
-   **Protocol 128**: Fully Satisfied (Harmonized Content).
-   **Protocol 101**: Enhanced (Security/Exclusion Integrity).

## Manifest Type Clarification

> **ADR 097 Update (2026-02-01):** Two distinct manifest categories exist:

| Category | Purpose | Schema | Tools |
|----------|---------|--------|-------|
| **Ingest Manifests** | Define scope for RAG/Forge ingestion | Directory-based (`ingest_manifest.json`) | `content_processor.py` |
| **Bundling Manifests** | Define files for context bundles | `{title, files: [{path, note}]}` | `bundle.py`, `manifest_manager.py` |

**Key Differences:**
- **Ingest**: Recursive directory traversal, exclusion filtering, syntax validation
- **Bundling**: Explicit file lists with annotations, no recursive traversal

**Do NOT confuse:**
- `mcp_servers/lib/ingest_manifest.json` (RAG/Forge scope) 
- `.agent/learning/*.json` (Protocol 128 bundling)

## Related Documents
-   [ADR 089: Modular Manifest Pattern for Context-Aware Snapshots](./089_modular_manifest_pattern.md)
-   [ADR 097: Base Manifest Inheritance Architecture](./097_base_manifest_inheritance_architecture.md)
-   [Protocol 130: Manifest Deduplication](../01_PROTOCOLS/130_Manifest_Deduplication_Protocol.md)
