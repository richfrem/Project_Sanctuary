---
name: learning-loop
description: "Protocol 128 Hardened Learning Loop for cognitive continuity across agent sessions. Use when: (1) starting any session (boot/orientation), (2) performing research or knowledge capture, (3) closing a session (seal, persist, retrospective), (4) user invokes /sanctuary-start, /sanctuary-learning-loop, /sanctuary-scout, /sanctuary-audit, /sanctuary-seal, /sanctuary-persist, /sanctuary-retrospective, /sanctuary-ingest, /sanctuary-end, or any Protocol 128 phase (I-X). Ensures knowledge is captured, validated, and persisted so the next session can resume from a sealed cognitive state."
---

# Learning Loop (Protocol 128)

Structured 10-phase cognitive continuity loop ensuring knowledge survives across agent sessions.

## The Iron Chain

Every session must **begin** with orientation and **end** with sealed persistence.

```
Scout → Synthesize → Gate → Audit → Seal → Persist → Retrospective → Ingest → End
```

## Quick Start: Session Open

**Mandatory**: Before starting, copy the meta-tasks from `.agent/templates/workflow/learning-loop-meta-tasks.md` into your task list.

1. Read boot files in order:
   ```
   .agent/learning/guardian_boot_contract.md    → Constraints
   .agent/learning/cognitive_primer.md          → Role
   .agent/learning/learning_package_snapshot.md → Context ("Cognitive Hologram")
   ```
2. Run `/sanctuary-scout` (Phase I)
3. Report readiness status

## Quick Start: Session Close

**Mandatory order:** Seal → Persist → Retrospective → End

```bash
/sanctuary-seal           # Phase VI  - Snapshot & validate
/sanctuary-persist        # Phase VII - Upload to HuggingFace
/sanctuary-retrospective  # Phase VIII - Self-reflection
/sanctuary-end            # Phase IX  - Git commit & closure
```

## Phase Reference

| Phase | Name | Command | Purpose |
|-------|------|---------|---------|
| I | Scout | `/sanctuary-scout` | Boot orientation & debrief |
| II | Synthesis | *(manual)* | Record ADRs, update LEARNING/ |
| III | Strategic Gate | *(HITL)* | Human reviews & approves |
| IV | Red Team Audit | `/sanctuary-audit` | Iterative research capture |
| V | RLM Synthesis | *(sovereign LLM)* | Generate cognitive hologram |
| VI | Seal | `/sanctuary-seal` | Snapshot + Iron Check |
| VII | Persist | `/sanctuary-persist` | HuggingFace upload |
| VIII | Self-Correction | `/sanctuary-retrospective` | Retrospective cycle |
| IX | Ingest & Close | `/sanctuary-ingest` → `/sanctuary-end` | RAG + git + closure |
| X | Phoenix Forge | *(optional)* | Fine-tuning from soul traces |

For detailed phase instructions, see [references/phases.md](references/phases.md).

## Pre-Departure Checklist

Before ending any session:

- [ ] Retrospective filled (`loop_retrospective.md`)
- [ ] Seal ran (`/sanctuary-seal`)
- [ ] Persist ran (`/sanctuary-persist`)
- [ ] Ingest ran (`/sanctuary-ingest`)
- [ ] Temp files cleaned: `rm -rf temp/context-bundles/*.md temp/*.md temp/*.json`

## Key Files

| File | Purpose |
|------|---------|
| `.agent/learning/cognitive_primer.md` | Role orientation |
| `.agent/learning/learning_package_snapshot.md` | Cognitive Hologram |
| `.agent/learning/guardian_boot_contract.md` | Immutable constraints |
| `.agent/learning/guardian_boot_digest.md` | Tactical status |
| `ADRs/071_protocol_128_cognitive_continuity.md` | Protocol ADR |
| `docs/architecture_diagrams/workflows/protocol_128_learning_loop.mmd` | Flow diagram |

## Infrastructure Prerequisites

Before running closure phases, check these services:

| Service | Needed For | Check | Skill |
|---------|-----------|-------|-------|
| **Ollama** | Phase VI (Seal) — RLM distillation | `curl -sf http://127.0.0.1:11434/api/tags` | `ollama-launch` |
| **ChromaDB** | Phase IX (Ingest) — RAG indexing | `curl -sf http://localhost:8110/api/v2/heartbeat` | `vector-db-launch` |

If either is offline, read the corresponding skill for startup instructions.

