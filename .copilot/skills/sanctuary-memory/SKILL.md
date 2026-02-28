---
name: sanctuary-memory
description: "Project Sanctuary-specific memory configuration. Maps the generic memory-management tiered system to Sanctuary's actual file paths, storage backends (RLM, Vector DB, Obsidian, HuggingFace), and persistence workflows."
---

# Sanctuary Memory Configuration

**Status:** Active
**Domain:** Project Sanctuary
**Depends on:** `memory-management` (generic tiered pattern), `rlm-factory`, `vector-db`, `obsidian-integration`, `huggingface-utils`

## Purpose

This skill maps the generic `memory-management` tiered architecture to Project Sanctuary's full storage stack. It knows every backend, file path, and plugin responsible for each memory tier.

## The Complete Memory Stack

```
┌─────────────────────────────────────────────────────────┐
│  HOT CACHE (always in context at boot)                  │
│  Files: .agent/learning/*                               │
│  ~200 lines total                                       │
├─────────────────────────────────────────────────────────┤
│  SEMANTIC CACHE (fast lookup, loaded on demand)         │
│  Backend: rlm-factory → rlm_summary_cache.json          │
│  Backend: rlm-factory → rlm_tool_cache.json             │
├─────────────────────────────────────────────────────────┤
│  VECTOR STORE (semantic search, loaded on demand)       │
│  Backend: vector-db → ChromaDB on port 8110             │
│  Profile: vector_profiles.json                          │
├─────────────────────────────────────────────────────────┤
│  DEEP STORAGE (filesystem, loaded on demand)            │
│  LEARNING/topics/, ADRs/, 01_PROTOCOLS/                 │
├─────────────────────────────────────────────────────────┤
│  VAULT (Obsidian, loaded on demand)                     │
│  Backend: obsidian-integration → OBSIDIAN_VAULT_PATH    │
│  Notes, canvases, graph connections                     │
├─────────────────────────────────────────────────────────┤
│  SOUL (external persistence, synced periodically)       │
│  Backend: huggingface-utils → HF Hub dataset            │
│  Repo: richfrem/Project_Sanctuary_Soul                  │
│  Structure: lineage/, data/, metadata/                  │
└─────────────────────────────────────────────────────────┘
```

## Tier 1: Hot Cache (Boot Files)

Loaded in order at every session start:

| Slot | Sanctuary File | Path |
|---|---|---|
| Primer | `cognitive_primer.md` | `.agent/learning/cognitive_primer.md` |
| Boot Digest | `guardian_boot_digest.md` | `.agent/learning/guardian_boot_digest.md` |
| Boot Contract | `guardian_boot_contract.md` | `.agent/learning/guardian_boot_contract.md` |
| Snapshot | `learning_package_snapshot.md` | `.agent/learning/learning_package_snapshot.md` |

**Target**: ~200 lines total across these 4 files.

## Tier 2: Semantic Cache (RLM Factory)

**Plugin**: `rlm-factory`
**Config**: `rlm_profiles.json`

| Cache | Purpose | Query Command |
|---|---|---|
| `rlm_summary_cache.json` | Code/doc summaries | `python plugins/rlm-factory/skills/rlm-curator/scripts/query_cache.py "keyword"` |
| `rlm_tool_cache.json` | Tool/script discovery | `python plugins/rlm-factory/skills/rlm-curator/scripts/query_cache.py --type tool "keyword"` |

**Refresh**: `python plugins/rlm-factory/skills/rlm-curator/scripts/distiller.py`

## Tier 3: Vector Store (ChromaDB)

**Plugin**: `vector-db`
**Config**: `vector_profiles.json`
**Server**: ChromaDB on `localhost:8110`

| Operation | Command |
|---|---|
| Query | `python plugins/vector-db/skills/vector-db-agent/scripts/query.py "semantic question"` |
| Ingest | `python plugins/vector-db/skills/vector-db-agent/scripts/ingest.py --profile default` |
| Cleanup | `python plugins/vector-db/skills/vector-db-agent/scripts/cleanup.py` |
| Launch server | Use `ollama-launch` skill equivalent for ChromaDB |

## Tier 4: Deep Storage (Filesystem)

| Slot | Sanctuary Location |
|---|---|
| Topics | `LEARNING/topics/{topic}/analysis.md` |
| Calibration | `LEARNING/calibration_log.json` |
| Decisions | `ADRs/{NNN}_{name}.md` (3-digit, via `adr-manager`) |
| Protocols | `01_PROTOCOLS/{NNN}_{name}.md` (via `protocol-manager`) |
| Chronicle | Journal entries (via `chronicle-manager`) |

## Tier 5: Vault (Obsidian)

**Plugin**: `obsidian-integration`
**Config**: `OBSIDIAN_VAULT_PATH` env var
**Guardian skill**: `sanctuary-obsidian-integration`

| Operation | Skill |
|---|---|
| Create/read/update/delete notes | `obsidian-vault-crud` |
| Parse markdown syntax | `obsidian-markdown-mastery` |
| Create visual diagrams | `obsidian-canvas-architect` |
| Traverse knowledge graph | `obsidian-graph-traversal` |
| Manage database views | `obsidian-bases-manager` |

## Tier 6: Soul (HuggingFace)

**Plugin**: `huggingface-utils`
**Guardian skill**: `sanctuary-soul-persistence`
**Dataset**: `richfrem/Project_Sanctuary_Soul`

| Operation | Function |
|---|---|
| Upload snapshot | `upload_soul_snapshot()` → `lineage/seal_<ts>_*.md` |
| Upload RLM cache | `upload_semantic_cache()` → `data/rlm_summary_cache.json` |
| Append traces | `append_to_jsonl()` → `data/soul_traces.jsonl` |
| Init structure | `ensure_dataset_structure()` → `lineage/`, `data/`, `metadata/` |

**Tags**: `project-sanctuary, cognitive-continuity, reasoning-traces, ai-memory, llm-training-data, metacognition`

## Memory Flow by Session Phase

### Boot (Phase I)
1. Load hot cache (Tier 1)
2. Iron check validates snapshot integrity
3. If stale → flag for refresh

### During Session
- **New learning** → Tier 4 (`LEARNING/topics/`)
- **Need context** → Tier 2 (RLM query) → Tier 3 (vector search) → Tier 4 (file read)
- **New decision** → `adr-manager` → Tier 4
- **New protocol** → `protocol-manager` → Tier 4
- **Journal entry** → `chronicle-manager` → Tier 4

### Closure (Phase VI-IX)
1. **Seal** → Update snapshot (Tier 1), capture state
2. **Persist** → Upload to HuggingFace (Tier 6)
3. **Ingest** → Refresh RLM cache (Tier 2) + Vector DB (Tier 3)
4. **Vault export** → Optionally write to Obsidian (Tier 5)
