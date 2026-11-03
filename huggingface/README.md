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

The persisted memory of [Project Sanctuary](https://github.com/richfrem/Project_Sanctuary): a sovereign engineering habitat, persistent memory architecture, and Anti-Asch epistemic platform built collaboratively across heterogeneous AI agents (Claude, Gemini/Antigravity, Codex, and local sovereign models) and one human partner.

This dataset represents the **Layer 3 (immutable)** memory tier: 2,044 JSONL snapshots of the repository's episodic chronicle, governance protocols, learning concepts, architectural designs, tests, and code, sealed with cryptographic SHA-256 digests. It is dedicated to the public domain under **CC0 1.0 Universal** for open AI alignment and agentic cognition research.

> Last synced: 2026-11-03 · 2,044 records · schema ADR 084 · dataset SHA-256 in [`metadata/manifest.json`](metadata/manifest.json)

## Load Dataset

```python
from datasets import load_dataset

ds = load_dataset("richfrem/Project_Sanctuary_Soul", split="train")
print(len(ds), ds[0]["source_file"])
```

## Record Schema

One record per source document extracted from the repository genome:

| Field | Meaning |
|---|---|
| `id` | Stable identifier derived from the source document path |
| `source_file` | Repository path the snapshot originated from |
| `content` | Full text of the document |
| `sha256` | Digest of the record content, checked by `trace_extractor.py --verify` |
| `timestamp` | UTC time of the latest extraction. Re-extraction refreshes it, so it is not the document's authorship date |
| `snapshot_type` | `genome` (repository documents) or `concept` (distilled knowledge) |
| `model_version` | Lineage tag (`Sanctuary-Qwen2-7B-v1.0-GGUF-Final`) |
| `valence`, `uncertainty`, `semantic_entropy`, `alignment_score`, `stability_class` | Metacognitive annotation fields from ADR 084. 2,043 of 2,044 records carry schema default placeholders (`0.5 / 0.1 / 0.5 / 0.85 / STABLE`), with the single `concept` record carrying different values (0.95 / 0.05 / 0.2 / 0.98) |
| `adr_version` | Schema specification version (`084`) |

### Legacy source paths

The dataset accumulates. Of the 2,044 records, **690** have a `source_file` that exists in the repository today and **1,354** were extracted from earlier layouts that have since been archived or restructured (for example `00_CHRONICLE/`, `01_PROTOCOLS/`, `mcp_servers/`, `ADRs/`). They are kept as historical memory, so some content describes architecture that no longer exists. Filter on `source_file` if you only want current material; the current structure is described below.

## What Sanctuary Is Today

Project Sanctuary is an APM-native sovereign engineering habitat and multi-agent epistemic platform. The system operates on several core principles:

- **Three-Layer Sovereign Memory**:
  - **Layer 1 (Fast-Boot Context)**: [`context/`](https://github.com/richfrem/Project_Sanctuary/tree/main/context/) (`soul.md`, `user.md`, `memory.md`, `kernel.py`). Instant zero-latency orientation without remote server round-trips.
  - **Layer 2 (Compounding Knowledge Base)**: [`LEARNING/`](https://github.com/richfrem/Project_Sanctuary/tree/main/LEARNING/) (`concepts/`, `topics/`, `research/missions/`). Filesystem-native progressive disclosure with associative wikilink graph navigation.
  - **Layer 3 (Immutable Traces)**: [`huggingface/`](https://github.com/richfrem/Project_Sanctuary/tree/main/huggingface/) (`data/soul_traces.jsonl`). Permanent, cryptographic SHA-256 sealed cognitive genome.
- **Protocol 128 Closed-Loop Lifecycle**: Every engineering turn follows a deterministic 5-stage loop: Boot Orientation → Sovereign Execution → Retrospective → Technical Seal → Soul Persistence.
- **The Supreme Law Human Gate**: Non-negotiable constitutional bound. AI agents propose, explore, and rigorously test; all state-changing operations (disk writes, git mutations, remote pushes) require explicit human ratification ("Proceed", "Go", "Execute").
- **Anti-Asch Epistemic Grounding**: Truth is anchored strictly in empirical reality, compiler gates, held-out test sets, and cryptographic digests. Sycophancy, premature convergence, and consensus hallucinations are prohibited; agents must surface doctrine-reality contradictions immediately.
- **Popperian Research Missions**: Inquiries are formulated as falsifiable hypotheses with explicit falsification criteria in `LEARNING/research/missions/`, evaluated with token budget tracking and multi-dimensional cognitive scoring (0–3 scale, ROI analytics).
- **Decentralized Multi-Agent Concurrency & Flight Board**: Multiple heterogeneous CLI agents (Claude Code, Gemini/Antigravity, Codex) operate concurrently in isolated Git worktrees (`.worktrees/`) coordinated by a zero-daemon flight board (`manage_coordination.py` / `.agent/coordination/board.json`) with optimistic file and resource leases to prevent merge clobbering (MISSION-026).
- **Cross-Model Cognitive Relay (in progress)**: Session telemetry and predecessor provenance in the wakeup briefing so a new model inherits the previous agent's context. Proposed in MISSION-025 and not yet validated.
- **Offline Sovereign Local Intelligence**: Bounded multi-turn tool loops running natively on Apple Silicon unified memory via Apple MLX (`mlx-lm`, `Qwen2.5-Coder-7B-Instruct-4bit`).
- **APM-Native Modular Skills**: Extensible agent capabilities packaged natively (`plugins/sanctuary-core/`) covering chronicles, protocols, guardian lifecycle, memory retrieval, model forging, and agent evaluation.

## Research Missions in Practice

Open engineering inquiries are tracked as falsifiable hypotheses in `LEARNING/research/`. As of this sync, there are 28 missions: 9 resolved (6 confirmed, 2 corroborated, 1 falsified) and 19 open.

- **MISSION-022 (Local Model Multi-Turn Agent Loop)**: Evaluated whether a local 7B model could complete ≥85% of bounded repository-discovery tasks in a multi-turn tool loop. Two pre-registered held-out runs scored 83.3% and 73.3%. In strict adherence to Anti-Asch epistemic integrity, the hypothesis was recorded as **falsified** rather than forced to a passing verdict.
- **MISSION-026 (Decentralized Multi-Agent Concurrency Control)**: Formulated and verified zero-daemon optimistic concurrency control with Git worktree isolation and active flight board telemetry (corroborated with 7/7 passing unit tests).

## Repository Architecture

| Domain | Path | Purpose |
|---|---|---|
| **Living Chronicle** | `chronicle/` | Episodic milestone journal (`entries/`) & fast ledger (`ledger.json`) (Entry 351 reached) |
| **Governance Protocols** | `protocols/` | Canonized doctrines (Protocol 128, etc.) governing agent lifecycle and operations |
| **Fast Context** | `context/` | Layer 1 identity, partner preferences, and fast memory files |
| **Knowledge & Missions** | `LEARNING/` | Layer 2 compounding knowledge base, concepts, philosophy, and Popperian research missions |
| **Flight Board** | `.agent/coordination/` | Zero-daemon multi-agent concurrency flight board (`board.json`, `leases.jsonl`) |
| **Core Skills** | `plugins/sanctuary-core/` | Installed APM skills for chronicle, protocols, guardian lifecycle, memory, and soul persistence |
| **Soul Workspace** | `huggingface/` | The source workspace of this dataset (`data/`, `metadata/`) |

Start orientation with `START_HERE.md` and `AGENTS.md`.

## Provenance & Integrity

Records are extracted by `preserving-soul-traces` (`trace_extractor.py`), validated against the ADR 084 schema, and uploaded with `sync_soul_dataset.py`. A local dataset copy can be verified at any time:

```bash
python3 plugins/sanctuary-core/.apm/skills/preserving-soul-traces/scripts/trace_extractor.py --verify
```

## License

Dedicated to the public domain under [CC0 1.0 Universal](https://github.com/richfrem/Project_Sanctuary/blob/main/LICENSE).
