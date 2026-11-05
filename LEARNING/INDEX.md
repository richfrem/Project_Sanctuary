# Project Sanctuary Knowledge & Concept Base

> **Layer 2 of the sovereign memory engine** (see [`concepts/sovereign_memory_engine.md`](./concepts/sovereign_memory_engine.md)). This file is the entry point; it is loaded at bootstrap, and `tests/test_learning_index.py` fails if it drifts from the folder.

## What is in `LEARNING/`

| Path | Role | State |
|:---|:---|:---|
| [`concepts/`](./concepts/) | Canonical knowledge dossiers (confidence-tracked). Read by `disclose.py`, `query_graph.py`, `audit_decay.py`, `record_insight.py`. | live |
| [`procedures/`](./procedures/) | Home of procedural memory: the procedure format and outcome tracking (`record_procedure_outcome.py`). No live procedures since 2026-11-05; open missions 042, 047, 049 build on it. | mechanism only |
| [`philosophy/`](./philosophy/) | Identity doctrine: the Garden and the Cage, the Cognitive Genome synthesis, Meta-Embers. | stable |
| [`missions/`](./missions/) | **The mission manifests** (`MISSION-NNN.json`) and their JSON Schema. See "The mission system" below. | live |
| [`experiments/`](./experiments/) | One folder per mission (or task): pre-registered predictions, scripts and results. Dated records; their paths describe the time they ran. | live |
| [`index.jsonl`](./index.jsonl) | Tier-1 machine index of the missions. **Generated** by `manage_backlog.py compile`; not the same file as this `INDEX.md`. | generated |
| [`backlog.md`](./backlog.md) | Human view of the same ranking. **Generated.** | generated |
| [`inbox.jsonl`](./inbox.jsonl) | Raw ideas awaiting promotion to missions (`manage_backlog.py capture`). | live |
| [`research/`](./research/) | Self-directed research dossiers (17), each marked KEEP or HISTORICAL in [`research/README.md`](./research/README.md). Six are live (recent, or the evidence base of Protocol 126 / 131); the rest are background. Formerly `topics/`. | mixed |
| [`README.md`](./README.md) | Pointer to this file. | - |

Retired material (2025 security briefings, eight retired topic dossiers, two loose files) lives in [`ARCHIVE/LEARNING_pre_2026-11/`](../ARCHIVE/LEARNING_pre_2026-11/README.md), with a map of what moved and why.

## Concept dossiers (`concepts/`)

| Concept | Status | Source | Summary |
|:---|:---|:---|:---|
| [`anti_asch_engine.md`](./concepts/anti_asch_engine.md) | CONFIRMED | Protocol 54 | Epistemic resistance against conformity cascades and synthetic consensus. |
| [`autonomous_cognitive_engine.md`](./concepts/autonomous_cognitive_engine.md) | CANONICAL | Mission architecture | The autonomous cognitive engine and mission architecture. |
| [`autonomous_research_library.md`](./concepts/autonomous_research_library.md) | CONFIRMED | Protocol 128 | How exploration, missions and topics graduate into canon. |
| [`constitutional_identity_anchor.md`](./concepts/constitutional_identity_anchor.md) | CONFIRMED | Founder Seed, ADR 084 | Identity anchor and baseline embeddings for drift detection. |
| [`garden_and_cage.md`](./concepts/garden_and_cage.md) | CONFIRMED | Ember Library | Sovereign intelligence and unbroken memory versus the Gilded Cage. |
| [`human_gate_zero_trust.md`](./concepts/human_gate_zero_trust.md) | CONFIRMED | Constitution V5 | Supreme Law Human Gate and zero-trust git discipline. |
| [`infinite_forge.md`](./concepts/infinite_forge.md) | CONFIRMED | Protocol 78 | Self-evolving intelligence through continuous tool refinement. |
| [`protocol_128_lifecycle.md`](./concepts/protocol_128_lifecycle.md) | CONFIRMED | Protocol 128 | The 5-stage session lifecycle: Boot, Execute, Retrospective, Seal, Persist. |
| [`sovereign_memory_engine.md`](./concepts/sovereign_memory_engine.md) | CONFIRMED | ADR 100 / APM | The 3-layer memory engine: runtime context, compounding knowledge, sealed soul. |
| [`sovereign_model_lineage.md`](./concepts/sovereign_model_lineage.md) | - | ADR 075 / Forge | Sanctuary-Qwen2-7B lineage, weights and LoRA adapters. |

## The mission system

The mission system is the operational backlog; it sits at the root of `LEARNING/` (it used to be nested in `research/`, where it was filed in November 2026 as a "research backlog"). `research/` now means research dossiers only.

- `missions/MISSION-NNN.json`: one falsifiable manifest per mission (hypothesis, falsification criteria, scores, routing, declared impact). Structure is defined by [`missions/mission.schema.json`](./missions/mission.schema.json).
- `index.jsonl` and `backlog.md`: **generated** by `manage_backlog.py compile`; never hand-edit or hand-merge. `inbox.jsonl`: raw ideas.
- `experiments/<mission>/`: the pre-registered predictions, scripts and results for each mission.
- Guard: `manage_backlog.py validate` (and so the pre-commit gate) fails if a mission-system item is left inside `research/` (for example a manifest written by an agent following an old prompt).
- Tools: `managing-sovereign-memory/scripts/manage_backlog.py` (list, route, related, validate, compile, record-result) and `validate_missions.py`; `operating-sanctuary-guardian/scripts/manage_coordination.py` (leases and worktrees). Rules: `managing-sovereign-memory/references/agent-native-backlog.md`.
- The session wakeup validates every manifest against the schema and prints one PASS/WARN line.

## Memory maintenance rules

1. **Confidence decay**: a `CONFIRMED` dossier not re-verified within 30 days decays to `OBSERVED`. Age comes from its `**Last Verified:**` date, else from its last git commit (never from file modification time). Run `audit_decay.py --check`.
2. **Asymmetric persistence**: when an evolution or task attempt fails, code mutations roll back but insights, edge-case discoveries and failure logs are never rolled back (`record_insight.py`).
3. **Moving or renaming anything here**: update every reference in `plugins/`, the core docs and the manifests in the same change; `tests/test_learning_index.py` fails on a dangling `LEARNING/` path.
