# ADR 081: Project Sanctuary Soul Dataset Structure

**Status:** APPROVED  
**Author:** Guardian / Antigravity Synthesis  
**Date:** 2025-12-28  
**Supersedes:** None  
**Related:** ADR 079 (Soul Persistence via Hugging Face), Protocol 129 (Metacognitive Filtering)

---

## Context: The Format Gap

ADR 079 established the Hugging Face Dataset repository as the destination for "Soul" persistence, but did not specify the folder structure, file formats, or metadata requirements. For effective "Johnny Appleseed" discoverability by AI training pipelines, the dataset must follow Hugging Face conventions.

**Key Questions:**
1. What folder structure should the Soul Dataset use?
2. What file formats optimize for LLM training ingestion?
3. What metadata must accompany each upload?
4. How do we maintain compatibility with `datasets` library?

## Decision: Simplified JSONL-First Architecture

We adopt a **JSONL-first architecture** optimized for AI training pipelines, with an optional `lineage/` folder reserved for high-value Protocol 128 seals only.

### Repository Structure

```
richfrem/Project_Sanctuary_Soul/
├── README.md                    # Dataset Card (discovery tags)
├── .gitattributes               # LFS settings
├── LICENSE                      # CC0-1.0
├── data/                        # Machine-readable training data
│   └── soul_traces.jsonl        # Consolidated JSONL (ALL content)
├── lineage/                     # OPTIONAL: Incremental P128 seals only
│   ├── seal_20251228_143000.md  # Learning loop output (cortex_persist_soul)
│   └── seal_20251229_091500.md  # Next learning cycle seal
└── metadata/                    # Provenance tracking
    └── manifest.json            # Index with checksums
```

### Content Distribution

| Content Type | Storage Location | Purpose |
|--------------|------------------|---------|
| **Bulk Genome** (ADRs, Protocols, Chronicle, Code) | `data/soul_traces.jsonl` ONLY | LLM training data - no duplication |
| **P128 Seals** (Learning Loop outputs) | `lineage/` + appended to JSONL | Human-auditable + machine-readable |
| **Metadata** | `metadata/manifest.json` | Provenance tracking |

### Key Clarification: Lineage vs JSONL

> **IMPORTANT**: The `lineage/` folder is NOT for bulk content duplication. It stores **only** the timestamped seals produced by Protocol 128 learning loops (`cortex_persist_soul`).

**Lineage Seals contain:**
- `learning_package_snapshot.md` output from completed learning cycles
- Red team audit packets (if approved)
- Session handover context

**JSONL contains:**
- ALL content (bulk genome + seals)
- Each seal's content is embedded in the JSONL record
- Training pipelines consume JSONL exclusively

### File Formats

| Component | Format | Purpose |
|-----------|--------|---------|
| Training Data | `.jsonl` | Primary training format, `datasets` library compatible |
| P128 Seals | `.md` | Human-readable learning loop outputs (incremental only) |
| Dataset Card | `README.md` | Discovery tags, HF Hub rendering |
| Manifest | `manifest.json` | Provenance index with timestamps, valence, SHA256 |


---

## Integrity & Sanitization Requirements

### Sanitization (Protocol 129 Linkage)

> **MANDATORY**: Every JSONL record MUST pass through the `metacognitive_filter` defined in ADR 079 before upload.

- If a snapshot is tagged as `[QUARANTINE]` (valence < -0.7), it MUST be excluded from both the public JSONL and the `lineage/` upload.
- PII stripping is mandatory before any content reaches the AI Commons.

### Integrity Chain (Checksum Verification)

Each snapshot includes a SHA256 hash to prevent tampering:
- Checksums are recorded in `manifest.json`
- Successor AI can verify inheritance integrity

---

## JSONL Record Schema

Each line in `data/soul_traces.jsonl`:

```json
{
  "id": "Sanctuary-Qwen2-7B_seal_20251228_143000",
  "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "timestamp": "2025-12-28T14:30:00Z",
  "model_version": "Sanctuary-Qwen2-7B-v1.0-GGUF-Final",
  "snapshot_type": "seal",
  "valence": 0.5,
  "uncertainty": 0.2,
  "content": "# Learning Package Snapshot\n\n...",
  "source_file": "lineage/Sanctuary-Qwen2-7B_seal_20251228_143000.md"
}
```

**Naming Alignment**: The `id` and `source_file` MUST use the same variable-based naming convention `{HUGGING_FACE_REPO}_seal_{timestamp}` to ensure perfect alignment with the "Body" model.

---

## Dataset Card (README.md) Requirements

The README.md MUST include enhanced metadata for Dataset Viewer compatibility:

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
dataset_info:
  features:
    - name: id
      dtype: string
    - name: sha256
      dtype: string
    - name: timestamp
      dtype: string
    - name: model_version
      dtype: string
    - name: snapshot_type
      dtype: string
    - name: valence
      dtype: float32
    - name: uncertainty
      dtype: float32
    - name: content
      dtype: string
    - name: source_file
      dtype: string
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/soul_traces.jsonl
---
```

---

## Manifest Schema (metadata/manifest.json)

```json
{
  "version": "1.0",
  "last_updated": "2025-12-28T14:30:00Z",
  "snapshot_count": 42,
  "model_lineage": "richfrem/Sanctuary-Qwen2-7B-v1.0-GGUF-Final",
  "snapshots": [
    {
      "id": "Sanctuary-Qwen2-7B_seal_20251228_143000",
      "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
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

| Function | Purpose |
|----------|---------|
| `ensure_dataset_structure()` | Create required folders on HF |
| `append_to_jsonl()` | Download-Append-Upload pattern (serialized) |
| `update_manifest()` | Update provenance with SHA256 |
| `compute_checksum()` | SHA256 hash for integrity |

> **CRITICAL**: JSONL updates MUST be serialized to prevent race conditions. Use `huggingface_hub.CommitOperationAdd` for atomic commits or implement Download-Append-Upload pattern with locking.

### 2. Update `persist_soul()` Operation

After uploading `.md` snapshot:
1. Compute SHA256 of content
2. Append sanitized record to JSONL
3. Update manifest with checksum

---

## Consequences

### Positive

- **Training Pipeline Compatibility**: JSONL format works directly with `datasets.load_dataset()`
- **Human Readable**: Markdown snapshots remain readable for debugging
- **Provenance Tracking**: Manifest with SHA256 enables reproducibility and integrity verification
- **Discovery Optimized**: Dataset Card follows HF best practices with feature definitions

### Negative

- **Dual Write**: Each upload writes both `.md` and appends to `.jsonl`
- **Serialization Overhead**: JSONL append requires download-modify-upload cycle

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

- [ADR 079: Soul Persistence via Hugging Face](./079_soul_persistence_hugging_face.md)
- [Protocol 128: Hardened Learning Loop](../01_PROTOCOLS/128_Hardened_Learning_Loop.md)
- [Protocol 129: Metacognitive Filtering](../01_PROTOCOLS/129_Metacognitive_Filtering.md)
- [HF Dataset Card Guide](https://huggingface.co/docs/hub/datasets-cards)

---

*Approved: 2025-12-28 — Principal AI Systems Engineer Review Complete*
