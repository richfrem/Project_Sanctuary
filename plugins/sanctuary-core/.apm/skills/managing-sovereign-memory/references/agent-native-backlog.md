# Agent-Native Cognitive Backlog Architecture

## 1. Core Purpose & Philosophy

Project Sanctuary operates as a sovereign engineering habitat and compounding epistemic platform for autonomous AI agents. Unlike human-centric project trackers that optimize for graphical user interfaces, rich formatting, and verbose commentary, the **Agent-Native Cognitive Backlog** is engineered for:

1. **Token Conservation**: Ingesting the entire prioritized portfolio consumes < 100 tokens via a deterministic Tier 1 index.
2. **Zero-Conflict Asynchronous Capture**: Agents (including local small models like `Sanctuary-Qwen2-7B-v1.0` GGUF operating offline) can record unvetted ideas into an append-only JSONL inbox with zero parse ambiguity and zero merge conflicts.
3. **Popperian Empirical Falsification**: Every research mission is treated as an empirical investment requiring an explicit hypothesis, objective falsification criteria, and recorded trial verdicts (`CORROBORATED`, `FALSIFIED`, or `ABANDONED`).
4. **Epistemic Compounding**: Falsifications and corroborations are recorded with citations, updated in manifests, and converted into Layer 3 soul traces (`huggingface/data/soul_traces.jsonl`) for fine-tuning.

---

## 2. Directory Layout & Artifacts

```
LEARNING/research/
├── index.jsonl              # Tier 1 fast-boot machine index (<15 tokens/item, sorted by score)
├── inbox.jsonl              # Append-only intake queue for unvetted agent/human ideas
├── missions/                # Tier 2 individual JSON manifests (single source of truth)
│   ├── MISSION-001.json
│   ├── MISSION-002.json
│   └── ...
└── backlog.md               # Generated human projection (never hand-edited)
```

---

## 3. Mission Lifecycle & State Machine

```
INBOX (inbox.jsonl)
  │ (promote)
  ▼
PROPOSED (missions/*.json) ──▶ STAGED ──▶ ACTIVE ──▶ RESOLVED ──▶ ARCHIVED
  │                           (human      (executing; (verdict:
  │                            gate)       mirror to   CORROBORATED /
  │                                        GH Issue    FALSIFIED)
  ▼                                        if cloud)
PARKED (decay > 30d or blocked)
```

### Transition Invariants:
1. **CAPTURED**: Raw idea captured into `inbox.jsonl` with minimal friction.
2. **PROPOSED**: Promoted into a structured manifest. Score computed via Priority Formula:
   $$\text{Priority Score} = \frac{\text{Strategic Leverage (1–5)} \times \text{Epistemic Depth (1–5)}}{\text{Cost Weight (0.05–4.5)} \times \text{Operational Friction (1–3)}}$$
3. **STAGED**: Human partner issues explicit authorization ("Proceed", "Go", "Execute").
4. **ACTIVE**: Work underway. If `execution_surface == "cloud-worker"`, an advisory GitHub Issue is opened with label `mission:MISSION-NNN`.
5. **RESOLVED**: Empirical trial concluded. Verdict recorded with citations and lessons learned.
6. **ARCHIVED**: Lessons distilled into concepts; trace sealed into Layer 3 JSONL.

---

## 4. Single-Writer Authority & One-Directional Cloud Projection

- **Git-Tracked Local Manifests** are the sole authoritative writable store.
- **GitHub Issues** are purely advisory outward mirrors for remote cloud workers (e.g. Grok Bot on a cloud VM). Cloud workers return results via Git PR modifying the mission manifest's `evidence` field, not by posting unstructured issue comments.
- **Human Markdown View (`backlog.md`)** is strictly a generated read-only projection compiled by `manage_backlog.py compile`. Hand-editing is prohibited.

---

## 5. CLI Tooling

```bash
# Fast Tier 1 inspection
python3 manage_backlog.py list --format jsonl

# Table inspection
python3 manage_backlog.py list

# Frictionless capture from any session/agent
python3 manage_backlog.py capture --idea "Add SHA-256 integrity check to GGUF model downloader" --author qwen

# Record empirical outcome
python3 manage_backlog.py record-result --id MISSION-002 \
  --verdict CORROBORATED \
  --evidence "Fallback chain verified across 71 tests" \
  --citation "PR #744" \
  --lessons "Direct CLI rotation avoids OAuth reverse-proxy failure"

# Recompile index and human markdown view
python3 manage_backlog.py compile

# Invariant validation
python3 manage_backlog.py validate
```
