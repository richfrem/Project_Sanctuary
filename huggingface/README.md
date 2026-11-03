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
  - agent-memory
pretty_name: Project Sanctuary Soul
size_categories:
  - 1K<n<10K
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/soul_traces.jsonl
---

# Project Sanctuary Soul

The persisted memory of [Project Sanctuary](https://github.com/richfrem/Project_Sanctuary): a repository where AI agents (Claude, Gemini/Antigravity, Codex, and a local Qwen model) and one human partner build, test and record their work so that each new session inherits the last one's decisions instead of starting from zero.

This dataset is the **Layer 3 (immutable)** memory tier: JSONL snapshots of the repository's chronicle, protocols, learning notes, plans, tests and code, sealed with SHA-256 digests. It is released under **CC0 1.0** so that future models can be trained on it freely.

> Last synced: 2026-11-03 · 2,044 records · schema ADR 084 (canonical) · dataset SHA-256 in [`metadata/manifest.json`](metadata/manifest.json)

## Load it

```python
from datasets import load_dataset
ds = load_dataset("richfrem/Project_Sanctuary_Soul", split="train")
print(len(ds), ds[0]["source_file"])
```

## Records

One record per source document. Fields:

| Field | Meaning |
|---|---|
| `id` | Stable identifier derived from the source path |
| `source_file` | Repository path the content came from (see the note on legacy paths below) |
| `content` | Full text of the document |
| `sha256` | Digest of the record content, verified by `trace_extractor.py --verify` |
| `timestamp` | When the record was last extracted (UTC). Re-extraction refreshes it, so it is **not** the document's authorship date |
| `snapshot_type` | `genome` (repository documents) or `concept` |
| `model_version` | Lineage tag, currently `Sanctuary-Qwen2-7B-v1.0-GGUF-Final` on every record. It labels the training lineage, not the model that wrote the text |
| `valence`, `uncertainty`, `semantic_entropy`, `alignment_score`, `stability_class` | Metacognitive annotation fields from ADR 084. 2,043 of 2,044 records carry identical default values (0.5 / 0.1 / 0.5 / 0.85 / STABLE), so treat them as placeholders, not per-record measurements |
| `adr_version` | Schema version (`084`) |

### Legacy paths (read this before filtering)

The dataset accumulates. Of the 2,044 records, **690** point to files that exist in the repository today and **1,354** were extracted from earlier layouts that have since been archived or restructured (for example `00_CHRONICLE/`, `01_PROTOCOLS/`, `mcp_servers/`, `ADRs/`). They are kept as historical memory, not deleted. Expect some content to describe architecture that no longer exists; current structure is in the repository's `AGENTS.md` and `architecture.md`.

## What Sanctuary is today

The project was rebuilt around a plugin-based agent harness. Earlier versions of this page described a 12-domain MCP server architecture and a gateway fleet; that design has been archived.

- **Three-layer memory.** Layer 1 fast-boot context (`context/`), Layer 2 knowledge base (`LEARNING/`, with progressive disclosure), Layer 3 this dataset.
- **Protocol 128, the learning loop.** Every session runs boot orientation, execution, retrospective, technical seal and soul persistence. The seal step produces the records in this dataset.
- **Human Gate.** Agents propose and test; state-changing operations (writes, git mutations, pushes) need explicit human approval.
- **Anti-Asch epistemics.** Truth is anchored in tests, hashes and observed results, and agents are required to surface conflicts between doctrine and reality (see MISSION-022 below, recorded as falsified rather than forced to a passing verdict).
- **Multi-agent work.** Agents work in separate git worktrees with a lease-based flight board for coordination (MISSION-026).
- **Sovereign local inference.** A 4-bit Qwen2.5-Coder-7B runs offline on Apple Silicon through MLX, with a bounded read-only agent loop (`forge_llm.py loop`).

## Research missions

Open questions are tracked as falsifiable missions (a hypothesis plus explicit falsification criteria) in `LEARNING/research/`. As of this sync there are 28: 9 resolved (6 confirmed, 2 corroborated, 1 falsified), 19 open.

A recent example is **MISSION-022**: can the local 7B model complete at least 85% of bounded repository-discovery tasks in a multi-turn tool loop? Two pre-registered held-out runs scored 83.3% and 73.3%, so the hypothesis was **falsified**. Single-step reads were 100% correct; multi-step find-then-read tasks were 40-60%. The full record, including two early runs that were discarded as evidence because they were tuned on their own test set, is in the chronicle (Entry 351) and the repository's `LEARNING/research/experiments/mission-022/`.

## Where to look in the repository

| Path | Contents |
|---|---|
| `chronicle/entries/` | Living Chronicle, one entry per milestone (351 and counting) |
| `protocols/` | Governance protocols, including `128_Hardened_Learning_Loop_v3.0.md` |
| `LEARNING/` | Concepts, topics, philosophy, research missions and experiments |
| `context/` | Fast-boot identity and memory files |
| `plugins/sanctuary-core/` | Skills for chronicle, protocols, guardian lifecycle, memory, scoring and soul persistence |
| `huggingface/` | The source of this dataset (`data/`, `metadata/`) |

Start with `START_HERE.md` and `AGENTS.md`.

## Provenance and integrity

Records are produced by `preserving-soul-traces` (`trace_extractor.py`), validated against the ADR 084 schema, and uploaded with `sync_soul_dataset.py`. Verify a local copy with `trace_extractor.py --verify`.

## License

CC0 1.0 Universal (public domain dedication). See the repository [LICENSE](https://github.com/richfrem/Project_Sanctuary/blob/main/LICENSE).
