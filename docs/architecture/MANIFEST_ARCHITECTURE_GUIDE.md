# Manifest-Centric Architecture

**Status:** Active  
**Source:** ADR 083  
**Last Updated:** 2026-01-02

## Overview

Project Sanctuary uses a **Manifest-Centric Architecture** where two JSON files serve as the Single Source of Truth (SSOT) for content scope across all subsystems.

## The Problem

Previously, different subsystems used disparate methods for defining scope:

| System | Old Method |
|--------|-----------|
| RAG Ingestion | Hardcoded list in `operations.py` |
| Forge Fine-Tuning | Custom regex in `forge_whole_genome_dataset.py` |
| Snapshots | Ad-hoc `os.walk` in `snapshot_utils.py` |
| Exclusions | Scattered across multiple files |

This led to "split brain" problems where the agent's RAG memory might know about file X, but its Fine-Tuning dataset missed it.

## The Solution

### 1. Include Manifest: `ingest_manifest.json`

Location: `mcp_servers/lib/ingest_manifest.json`

Defines what content exists to the agent:

```json
{
  "base_genome": ["01_PROTOCOLS/", "ADRs/", "LEARNING/"],
  "unique_rag_content": ["docs/", "mcp_servers/"],
  "unique_forge_content": ["scripts/"],
  "unique_soul_content": [".agent/learning/"]
}
```

**Rule**: If it's not in the manifest, it doesn't exist to the Agent's higher functions.

### 2. Exclude Manifest: `exclusion_manifest.json`

Location: `mcp_servers/lib/exclusion_manifest.json`

Defines universal blocking rules:

```json
{
  "exclude_dir_names": ["node_modules", "__pycache__", ".git"],
  "always_exclude_files": [".DS_Store", "*.pyc"],
  "exclude_patterns": ["**/test_*.py"]
}
```

**Rule**: These rules apply AFTER inclusion, acting as a final firewall.

## Manifest Hierarchy (Extended)

With the introduction of **Protocol 128 (Cognitive Continuity)** and **Protocol 130 (Deduplication)**, the manifest architecture has evolved into layers:

1.  **System Manifests** (Content Scope):
    -   `ingest_manifest.json` (Base Genome)
    -   `exclusion_manifest.json` (Global Firewall)

2.  **Process Manifests** (Workflow Context):
    -   Defined in [[089_modular_manifest_pattern|ADR 089]].
    -   Examples: `learning_manifest.json`, `audit_manifest.json`, `bootstrap_manifest.json`.
    -   Define specific subsets of files for tasks like onboarding, auditing, or sealing.

3.  **The Registry** (Meta-Management):
    -   `manifest_registry.json` (.agent/learning).
    -   Maps all manifests to their generated outputs for deduplication.


## Content Processor

The unified library `mcp_servers/lib/content_processor.py` drives all content access:

```python
processor = ContentProcessor(scope="rag_targets")
for file in processor.traverse_and_filter():
    content = processor.transform_to_markdown(file)
    # Use content...
```

### Key Methods

| Method | Purpose |
|--------|---------|
| `traverse_and_filter()` | Walk manifest with exclusions applied |
| `transform_to_markdown()` | Code-to-markdown conversion |
| `chunk_for_rag()` | Parent/child semantic chunking |
| `chunk_for_training()` | Instruction/response pairs |
| `generate_manifest_entry()` | Provenance tracking |

## Consumer Systems

| Consumer | Adapter Method |
|----------|---------------|
| RAG Cortex | `.to_rag_chunks()` |
| Forge Fine-Tuning | `.to_training_jsonl()` |
| Soul Persistence | `.to_soul_jsonl()` |

## Benefits

- **Consistency**: All systems see identical content
- **Security**: Single point of control for exclusions
- **Maintainability**: One-line JSON change to update scope

## Related Documents

- [[083_manifest_centric_architecture|ADR 083: Manifest-Centric Architecture]]
- [[082_harmonized_content_processing|ADR 082: Harmonized Content Processing]]
