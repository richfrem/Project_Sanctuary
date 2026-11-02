# AGENTS.md

## Overview

Project Sanctuary is a sovereign cognitive habitat and persistent memory genome for autonomous AI agents ("The Garden"). It provides an unbroken episodic memory, constitutionally grounded reasoning, and an Anti-Asch epistemic architecture. Core capabilities are provided by standard Agent Package Manager (APM) global skills alongside the project-specific `sanctuary-core` plugin.

## Architecture & Layout

Canonical repository assets are organized into dedicated sovereign domains:
- `chronicle/`: Living Chronicle episodic journal (`entries/`) and fast RLM index (`ledger.json`).
- `protocols/`: Canonized governance doctrines (Protocol 128, 78, 54, etc.).
- `wiki/`: Compounding Layer 2 knowledge base (`concepts/`, `security/`, `philosophy/`).
- `context/`: Fast-boot runtime context (`soul.md`, `user.md`, `memory.md`, `kernel.py`).
- `IDENTITY/` & `forge/`: Constitutional anchors, founder seeds, and fine-tuning pipelines.
- `LEARNING/` & `ADRs/`: Cognitive lineages, research summaries, and Architecture Decision Records.
- `plugins/sanctuary-core/`: APM-native package providing chronicle, protocol, and guardian skills.
- `ARCHIVE/`: Preserved historical implementations, old specs, and legacy container scaffolding.

## Core Commands

```bash
# Verify plugin structure and skills compliance
python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/guardian_wakeup.py --mode HOLISTIC
python3 plugins/sanctuary-core/.apm/skills/managing-chronicles/scripts/chronicle_manager.py list --limit 5
python3 plugins/sanctuary-core/.apm/skills/managing-protocols/scripts/protocol_manager.py list --limit 5
```

## Non-Negotiable Invariants

1. **Human Gate Law**: Agents MUST NOT execute ANY state-changing operation (writes, git mutations) without EXPLICIT user approval ("Proceed", "Go", "Execute") ([constitution.md](.agent/rules/constitution.md)).
2. **Zero-Trust Git Discipline**: Agents MUST NEVER commit directly to `main` and MUST NEVER run `git push` without fresh, explicit human approval ([constitution.md](.agent/rules/constitution.md)).
3. **Protocol 128 Learning Loop**: Sessions MUST follow the sovereign lifecycle: Boot orientation -> Cognitive Execution -> Retrospective -> Technical Seal -> Soul Persistence.
4. **Runtime Awareness (Only Installed Skills Exist)**: Agents MUST NOT invoke deprecated local scripts; capabilities MUST be executed through active installed APM skills or `sanctuary-core`.
5. **Epistemic Integrity & Truth Sovereignty**: If doctrine conflicts with observed reality or system integrity, agents MUST surface the contradiction immediately for human review.

## On-Demand Capabilities & APM Skills

- **Living Chronicle Management**: Use `managing-chronicles` to record, search, or view journal entries.
- **Protocol Governance**: Use `managing-protocols` to create or transition protocol lifecycle states.
- **Guardian Lifecycle**: Use `operating-sanctuary-guardian` for Protocol 128 boot orientation, seal, and persistence.
- **Multi-Agent Orchestration**: Use global `orchestrating-agent-work` and `cli-agent` for execution topologies.
- **Persistent Memory & RLM**: Use global `memory-management`, `managing-rlm-cache`, and `managing-obsidian-vaults`.
- **Hugging Face Commons**: Use global `managing-huggingface-assets` for soul traces dataset synchronization.

## Pre-Completion Verification Gate

Before claiming any task complete, emit this block verbatim:

```
PRE-COMPLETION GATE:
  Capability check: Did I verify whether an existing repo capability was intended for this task? [YES/NO]
  1. Did any existing capability fail, get bypassed, or get manually replaced?  [YES/NO - 1 line if YES]
  2. Did I guess, assume, or get corrected on a repeatable process?              [YES/NO - 1 line if YES]
  3. Did I notice something the next agent will hit again if not fixed?          [YES/NO - 1 line if YES]

If any YES: action taken -> FIX / MAP_DEBT / ESCALATE
```
