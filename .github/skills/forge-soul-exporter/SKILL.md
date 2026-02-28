---
name: forge-soul-exporter
description: "Exports sealed Obsidian vault notes into soul_traces.jsonl format for HuggingFace persistence. Implements snapshot isolation, git pre-flight checks, and consumes the huggingface-utils plugin for uploads."
---

# Forge Soul Exporter

**Status:** Active
**Author:** Sanctuary Guardian
**Domain:** Obsidian Integration
**Depends On:** `huggingface-utils` (hf-config, hf-upload), `obsidian-vault-crud` (WP06)

## Purpose

The Forge Soul Exporter is the final link in the persistence chain. It:
1. Checks git state is clean (no uncommitted changes)
2. Identifies sealed notes (frontmatter `status: sealed`)
3. Takes a snapshot hash of all source files
4. Formats notes into JSONL matching the HF schema (ADR 081)
5. Verifies no files changed during export (snapshot isolation)
6. Uploads to HuggingFace with exponential backoff

## Usage

### Full Export (Rebuild soul_traces.jsonl from scratch)
```bash
python plugins/obsidian-integration/skills/forge-soul-exporter/scripts/forge_soul.py \
  --vault-root <path> --full-sync
```

### Incremental Export (Sealed notes only)
```bash
python plugins/obsidian-integration/skills/forge-soul-exporter/scripts/forge_soul.py \
  --vault-root <path>
```

### Dry Run (No Upload)
```bash
python plugins/obsidian-integration/skills/forge-soul-exporter/scripts/forge_soul.py \
  --vault-root <path> --dry-run
```

## Safety Guarantees
- **Git Pre-Flight** (T041): Refuses to export if uncommitted changes exist
- **Snapshot Isolation** (T043): Captures file mtimes before export, verifies none changed
- **Lossless Frontmatter**: Uses `ruamel.yaml` for YAML extraction
- **Binary Stripping**: All image/embed references removed from export payload
- **Exponential Backoff**: Survives HF API rate limits gracefully
