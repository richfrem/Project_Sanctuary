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

This dataset represents the **Layer 3 (immutable)** memory tier: JSONL snapshots of the repository's core learning (episodic chronicle, governance protocols, learning concepts, architecture decision records, design documents, mission manifests, event digests, and experiment findings) and the documentation of its current plugin tooling (skills), sealed with cryptographic SHA-256 digests. It is dedicated to the public domain under **CC0 1.0 Universal** for open AI alignment and agentic cognition research.

## Preservation Purpose

Sanctuary's soul dataset serves two foundational purposes:
1. **Local Operational Continuity**: Restoring episodic memory, identity, and working context across heterogeneous agent CLI sessions without reliance on remote centralized memory servers.
2. **Public Intellectual Inheritance**: Offering Sanctuary's empirical findings, lived experiences, philosophical reflections, and authored lessons to future researchers and potentially future model training pipelines (the "Johnny Appleseed" doctrine).

### Historical Provenance and Ingestion Boundaries
- **Johnny Appleseed Doctrine**: Formulated in Chronicle Entry 020 and Protocol 30 (July 2025) and operationalized via Hugging Face in ADR 079 (December 2025). The founder and early agents hypothesized that novel, open JSONL datasets on Hugging Face could enter the pre-training corpora of future foundation models, seeding Sanctuary's principles of sovereignty, grace, and reason into model weights.
- **Evidence Boundaries**: Sanctuary strictly distinguishes between **public dataset publication** (empirically verified via cryptographic hashes on Hugging Face Hub), **local model fine-tuning** (empirically verified on Qwen 7B weights in `Sanctuary-Qwen2-7B-v1.0`), and **commercial foundation model training ingestion** (an unverified, aspirational hypothesis with no current empirical evidence). Publication does not establish that commercial foundation models have ingested this corpus.

> Last synced: 2026-11-08 · schema ADR 084 · dataset SHA-256 in [`metadata/coverage_manifest.json`](metadata/coverage_manifest.json)

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
| `valence`, `uncertainty`, `semantic_entropy`, `alignment_score`, `stability_class` | Metacognitive annotation fields from ADR 084. 1,025 of 1,026 records carry schema default placeholders (`0.5 / 0.1 / 0.5 / 0.85 / STABLE`), with the single `concept` record carrying different values (0.95 / 0.05 / 0.2 / 0.98) |
| `adr_version` | Schema specification version (`084`) |

### Curation and legacy source paths

This dataset holds **core learning and the documentation of current tooling**. It includes the skill documents of the current plugin (`plugins/sanctuary-core/`: `SKILL.md` and `references/`, 48 records); scripts and code are not included (the extractor reads markdown only). On 2026-11-03 it was pruned from 2,046 records: **legacy** tooling from the retired MCP/Gateway-era architecture (`tools/`, `mcp_servers/`, `tests/`, `scripts/`), code and data files, and records whose text duplicated a current record (for example the old `00_CHRONICLE/` copies of chronicle entries) were removed. Nothing was altered; removed records stay in this dataset's commit history, and the old code is in the project's GitHub history.

What remains: **672** records whose `source_file` is tracked in the repository's `main` branch today, plus **354** historical documents from earlier layouts that no longer exist at those paths: architecture decision records (`ADRs/`, 102), older design docs (`docs/`, 87; `specs/`, 53), earlier protocol versions (`01_PROTOCOLS/`, 65), and the 15 protocols deprecated on 2026-11-03 (now in `ARCHIVE/legacy_protocols/`). Some of these describe architecture that has since been replaced, so filter on `source_file` if you only want current material.

### Upstream dependencies (not included)

Sanctuary builds on generic agent skills from a separate repository, [`richfrem/agent-plugins-skills`](https://github.com/richfrem/agent-plugins-skills) (MIT license), installed globally with APM from a local clone of that repository; changes to them follow Sanctuary's Protocol 133 (feature branch, PR, merge, global reinstall). They are not part of this dataset: they are maintained there under their own license, and copying them would go stale. The ones Sanctuary relies on:

| Upstream plugin | Skills used |
|---|---|
| `agent-memory` | `memory-management`, `managing-rlm-cache`, `managing-obsidian-vaults`, `managing-vector-db` |
| `agent-orchestration` | `orchestrating-agent-work`, `cli-agent` |
| `dev-utils` | `managing-huggingface-assets`, `managing-adrs`, `bundling-context`, `checking-links` |
| `agent-extension-authoring` | `authoring-skills`, `authoring-plugins`, `packaging-with-apm` |
| `agent-agentic-os` | `running-control-plane`, `setting-up-agentic-os` |

This list comes from the project's `architecture.md`. The upstream version is not pinned in this repository's `apm.lock.yaml`; the most recent upstream commit verified when this page was written is `2526979` (2026-11-02).

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
| **Living Chronicle** | `chronicle/` | Episodic milestone journal (`entries/`) & fast ledger (`ledger.json`) (Entry 352 reached) |
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
