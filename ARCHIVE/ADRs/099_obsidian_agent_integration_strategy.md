---
title: "ADR 099: Obsidian Agent Integration Strategy"
status: "Proposed"
date: "2026-02-27"
tags: ["obsidian", "architecture", "agents", "integration"]
---

# ADR 099: Obsidian Agent Integration Strategy & Capability Boundary

## Status

Proposed

## Context & Problem Statement

Project Sanctuary needs to tightly integrate its autonomous agents with Obsidian Vaults to support living knowledge bases, cognitive continuity, and dynamic visualization (JSON Canvas, Obsidian Bases). We need a defined integration strategy that maps Obsidian's native capabilities against our existing agent-loop architecture without introducing redundant maintenance overhead.

This ADR addresses three critical subtasks (from WP01):
1. **Integration Mechanisms**: Should agents communicate with Obsidian via a Local REST API plugin, a custom TypeScript plugin, or direct Python `pathlib`/`frontmatter` operations?
2. **Capability Overlap**: How do Obsidian’s native semantic features compare to our existing `rlm-factory` and `vector-db` Python skills, and how do we avoid reinventing the wheel?
3. **Agent Skills & Boundaries**: What are the specific skill boundaries required to parse Obsidian-flavoured Markdown, `.base` data tables, and `.canvas` files?

## Decision

We will adopt a **Direct Filesystem Read/Write (Zero-RPC) Architecture** using Python hooks. We will explicitly reject Local REST API servers and custom TypeScript Obsidian plugins in favor of treating the Obsidian Vault as a standard, flat directory of Markdown, YAML, and JSON files manipulated by isolated Python skills.

### 1. Integration Mechanism: Direct Filesystem (Zero-RPC)
- **Why rejected REST/TS**: Building a TS obsidian plugin or relying on a 3rd party Local REST API requires the Obsidian desktop application to be actively running to process agent requests. It introduces cross-language maintenance (TS + Python), complex IPC, and brittle state matching.
- **Why chosen Direct Python**: Project Sanctuary agents already operate on local Git worktrees and filesystems. Parsing `.md` files with YAML frontmatter via Python's `pathlib` and `ruamel.yaml` natively aligns with our existing Git Worktree workflow (the `spec-kitty` implement protocol).

### 2. Capability Overlap (Native vs Custom)
- **Obsidian Graph / Search**: Obsidian natively excels at generic visual graph traversal and full-text search.
- **Project Sanctuary semantic skills**: Our `rlm-factory` (Semantic Ledger) and `vector-db` (ChromaDB embeddings) generate LLM-optimized summaries and semantic distances that Obsidian cannot natively compute without external cloud LLM plugins.
- **Boundary**: We will treat Obsidian as the *Presentation and Authoring Layer*. The `rlm-factory` and `vector-db` will remain the *Cognitive Retrieval Layer*. Agents will write to Obsidian files; Obsidian will visualize them. When agents need semantic retrieval, they will still query `vector-db`, not Obsidian's search.

### 3. Agent Skill Boundaries
To support direct filesystem manipulation, we will scaffold three pristine Python skills:
1. **Obsidian Markdown Mastery**: A shared Python parser for wikilinks (`[[Note|Alias]]`), block references (`^block`), and Obsidian callouts (`>[!NOTE]`).
2. **Obsidian Bases Manager**: A tool to structurally read/write `.base` files (or Markdown tables) preserving schema.
3. **JSON Canvas Architect**: A programmatic engine for generating Obsidian Node/Edge `.canvas` files to visualize agent reasoning traces seamlessly.

**Prerequisites**: The only host software required is Obsidian itself (for human viewing) and standard Python 3.12 (for agent operations).

## Consequences

*   **Positive**:
    *   Zero dependencies on the Obsidian desktop app being open or running background servers.
    *   Total parity with our existing "isolated worktree" execution model (`spec-kitty implement`).
    *   Agents can modify notes offline, concurrently, and inside isolated test vaults.
*   **Negative / Risks**:
    *   Requires writing robust, Obsidian-flavored Markdown parsers in Python (Wikilinks, embeds, block-refs).
    *   Must handle file-locking and atomic writes carefully to prevent corrupting files while a human is simultaneously typing in the Obsidian UI.

## Related ADRs
- Supports **ADR 081** (Soul Dataset Structure) by cleanly decoupling the source data from the presentation layer.
- Builds upon **ADR 094** (Soul Persistence of Semantic Cache) by keeping vector mappings inside our standalone plugins.
