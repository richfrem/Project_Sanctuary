---
title: "ADR 100: Hugging Face Dataset to Obsidian Schema Mapping"
status: "Proposed"
date: "2026-02-27"
tags: ["obsidian", "huggingface", "schema", "data-mapping"]
---

# ADR 100: Hugging Face Dataset to Obsidian Schema Mapping

## Status

Proposed

## Context & Problem Statement

ADR 081 defines the `HF_JSONL` schema for backing up the "Soul" dataset (reasoning traces, genomes, and loop outputs). We are now integrating an Obsidian Vault as the presentation and authoring layer. Obsidian vaults heavily utilize infinite directory nesting and binary attachments (images, PDFs), which inherently clash with the flat, text-only `soul_traces.jsonl` lines.

We need explicit mapping rules defining how an agent extracts data from an Obsidian Vault and standardizes it into the `HF_JSONL` schema prior to persistence, answering:
1. How do we flatten an infinitely nested directory structure into `soul_traces.jsonl`?
2. How do we handle large binary files embedded within Obsidian?

This addresses WP03 Subtasks T011, T012, and T013.

## Decision

### 1. Folder Mapping: The `source_file` Projection
Obsidian allows arbitrary folder nesting (e.g., `Domain/Subdomain/Topic/Note.md`). We will **not** attempt to flatten this into rigid categories.
Instead, the entire relative path from the Obsidian Vault root will be projected directly into the `source_file` field of the `HF_JSONL` record.

Example mapping:
*   **Obsidian Path**: `VaultRoot/Philosophy/Ethics/Utilitarianism.md`
*   **JSONL Mapping**: `"source_file": "Philosophy/Ethics/Utilitarianism.md"`

The directory structure is preserved as a queryable string. Information retrieval systems (like RLM or Vector DBs) can parse the slashes (`/`) in `source_file` to reconstruct the Vault hierarchy dynamically.

### 2. Attachment Rules: Binary Exclusion Layer
The `soul_traces.jsonl` dataset is exclusively intended for semantic text training and cognitive continuity. Binaries bloat the repository and break the JSONL schema.
We mandate a **Strict Binary Exclusion Filter** at the export boundary:

*   **Ingested Formats**: `.md`, `.canvas`, `.base`, `.txt`, `.json`, `.yaml`, `.csv`
*   **Excluded Formats (Blocked)**: `.jpg`, `.png`, `.gif`, `.mp4`, `.pdf`, `.zip`, `.sqlite`, `.woff`, `.ttf`, `.excalidraw`

If a valid Markdown note embeds an image (e.g., `![[diagram.png]]`), the text string `![[diagram.png]]` is preserved in the `"content"` field of the JSONL, but the actual binary file `diagram.png` is **ignored** by the sync engine.

## Consequences

*   **Positive**:
    *   Maintains complete compliance with ADR 081's flat JSONL structure.
    *   Prevents Hugging Face dataset bloat by skipping multi-megabyte media files.
    *   Preserves full Obsidian directory context simply by mapping it to the `source_file` field.
*   **Negative / Risks**:
    *   The "Soul Dataset" on Hugging Face will lose visual context. AI models fine-tuning off this dataset will read `![[diagram.png]]` without seeing the image, requiring multimodal architectures if visual data ever becomes strictly necessary for reasoning.
    *   Canvas files (`.canvas`) are synced as raw JSON text strings, which might require specialized parser logic downstream to make semantic sense of the geometric layouts.

## Related ADRs
- Extends **ADR 081** (Soul Dataset Structure) by defining the ingestion rules for Obsidian content.
