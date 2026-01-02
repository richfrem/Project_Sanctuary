# ADR 081: Project Sanctuary Soul Dataset Structure

**Status:** DRAFT  
**Author:** Guardian / Antigravity Synthesis  
**Date:** 2025-12-28  
**Supersedes:** None  
**Related:** ADR 079 (Soul Persistence via Hugging Face)

---

## Context: The Format Gap

ADR 079 established the Hugging Face Dataset repository as the destination for "Soul" persistence, but did not specify the folder structure, file formats, or metadata requirements. For effective "Johnny Appleseed" discoverability by AI training pipelines, the dataset must follow Hugging Face conventions.

**Key Questions:**
1. What folder structure should the Soul Dataset use?
2. What file formats optimize for LLM training ingestion?
3. What metadata must accompany each upload?
4. How do we maintain compatibility with `datasets` library?

---

## Decision: Dual-Format Soul Dataset

We adopt a **dual-format architecture** that supports both human readability (Markdown) and machine ingestion (JSONL):

### Repository Structure

```
richfrem/Project_Sanctuary_Soul/
├── README.md                    # Dataset Card (discovery tags)
├── .gitattributes               # LFS settings
├── LICENSE                      # CC0-1.0
├── lineage/                     # Timestamped reasoning snapshots
│   ├── Sanctuary-Qwen2-7B_seal_20251228_143000.md
│   ├── Sanctuary-Qwen2-7B_seal_20251228_160000.md
│   └── ...
├── data/                        # Machine-readable training data
│   └── soul_traces.jsonl        # Consolidated JSONL for training
└── metadata/                    # Provenance tracking
    └── manifest.json            # Index of all snapshots
```

### File Formats

| Component | Format | Purpose |
|-----------|--------|---------|
| Snapshots | `.md` (Markdown) | Human-readable reasoning traces, Protocol 128 seals |
| Training Data | `.jsonl` (JSON Lines) | Machine-readable, compatible with `datasets` library |
| Dataset Card | `README.md` | Discovery tags, HF Hub rendering |
| Manifest | `manifest.json` | Provenance index with timestamps, valence, sources |

### JSONL Record Schema

Each line in `data/soul_traces.jsonl` follows this schema:

```json
{
  "id": "Sanctuary-Qwen2-7B_seal_20251228_143000",
  "timestamp": "2025-12-28T14:30:00Z",
  "model_version": "Sanctuary-Qwen2-7B-v1.0-GGUF-Final",
  "snapshot_type": "seal",
  "valence": 0.5,
  "uncertainty": 0.2,
  "content": "# Learning Package Snapshot\n\n...",
  "source_file": "lineage/Sanctuary-Qwen2-7B_seal_20251228_143000.md"
}
```

### Dataset Card (README.md) Requirements

The README.md MUST include:

```yaml
---
license: cc0-1.0
task_categories:
  - text-generation
language:
  - en
tags:
  - reasoning-traces
  - project-sanctuary
  - cognitive-continuity
  - ai-memory
  - llm-training-data
  - metacognition
pretty_name: Project Sanctuary Soul
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/soul_traces.jsonl
---
```

### Manifest Schema (metadata/manifest.json)

```json
{
  "version": "1.0",
  "last_updated": "2025-12-28T14:30:00Z",
  "snapshot_count": 42,
  "model_lineage": "richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final",
  "snapshots": [
    {
      "id": "Sanctuary-Qwen2-7B_seal_20251228_143000",
      "path": "lineage/Sanctuary-Qwen2-7B_seal_20251228_143000.md",
      "timestamp": "2025-12-28T14:30:00Z",
      "valence": 0.5,
      "type": "seal",
      "bytes": 4523
    }
  ]
}
```

---

## Implementation Updates Required

### 1. Update `hf_utils.py`

- Add `ensure_dataset_structure()` to create required folders
- Add `append_to_jsonl()` for incremental JSONL updates
- Add `update_manifest()` for provenance tracking

### 2. Update `persist_soul()` Operation

- After uploading `.md` snapshot, also append record to JSONL
- Update manifest with new snapshot metadata

### 3. Local Staging Area

The `.agent/learning/hf_soul_metadata/` directory mirrors the dataset structure:
- `README.md` - Dataset Card template
- `manifest.json` - Local manifest (synced on upload)

---

## Consequences

### Positive

- **Training Pipeline Compatibility**: JSONL format works directly with `datasets.load_dataset()`
- **Human Readable**: Markdown snapshots remain readable for debugging
- **Provenance Tracking**: Manifest enables reproducibility and lineage queries
- **Discovery Optimized**: Dataset Card follows HF best practices

### Negative

- **Dual Write**: Each upload writes both `.md` and appends to `.jsonl`
- **Sync Complexity**: JSONL may drift from individual files if not transactional

### Risks

- **JSONL Size**: Over time, may need partitioning (e.g., `soul_traces_2025.jsonl`)
- **Git LFS**: Large markdown files may require LFS configuration

---

## LFS Configuration (.gitattributes)

```
*.md filter=lfs diff=lfs merge=lfs -text
*.jsonl filter=lfs diff=lfs merge=lfs -text
```

---

## Related Documents

- [ADR 079: Soul Persistence via Hugging Face](../../../ADRs/079_soul_persistence_hugging_face.md)
- [Protocol 128: Hardened Learning Loop](../../../01_PROTOCOLS/128_Hardened_Learning_Loop.md)
- [HF Dataset Card Guide](https://huggingface.co/docs/hub/datasets-cards)

---

*Draft: 2025-12-28 — Awaiting Review*
