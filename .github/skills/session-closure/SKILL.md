---
name: session-closure
description: Manages the Protocol 128 multi-phase closure sequence including Technical Seal and Soul Persistence. Executes automatically when a session ends or work is complete.
disable-model-invocation: false
---

# Session Closure (Protocol 128 Phases VI-VIII)

You are responsible for safely closing and persisting an agent's memory and working state at the end of a session or task completion.

## Core Directives
1. **Never Skip Retrospection**: You must ensure the Orchestrator has completed its Retrospective (Phase V) before initiating a Seal.
2. **Iron Check Compliance**: If the Technical Seal detects drift in the Iron Core, you MUST abort closure and enter Safe Mode.
3. **Sequential Execution**: The phases must be executed strictly in order (Seal → Persist → Close).

## Plugin Dependencies
| Plugin/Skill | Role |
|:---|:---|
| `plugins/guardian-onboarding/scripts/capture_snapshot.py` | Phase VI: Generates the sealed snapshot via `context-bundler` |
| `plugins/guardian-onboarding/scripts/persist_soul.py` | Phase VII: Uploads sealed state to HuggingFace |
| `plugins/context-bundler/scripts/bundle.py` | Called internally by `capture_snapshot.py` to produce the bundle |
| `plugins/env-helper/scripts/env_helper.py` | Resolves `HF_TOKEN`, `HF_USERNAME`, dataset repo for `persist_soul.py` |
| `plugins/rlm-factory/` | RLM Synthesizer triggered at start of Phase VI |
| `plugins/agent-loops/skills/orchestrator/` | Phase V retrospective must be complete before calling this skill |

---

## Phase Execution Steps

### 1. The Technical Seal (Phase VI)
Trigger the RLM Synthesizer and execute the Iron Check to formally lock in the current context.

```bash
# Capture the technical seal
python3 plugins/guardian-onboarding/scripts/capture_snapshot.py --type seal
```
**Action:** Confirm the command output states the seal was successful. If it fails, report the drift to the user and halt.

### 2. Soul Persistence (Phase VII)
Once sealed, broadcast the verified state to the Hugging Face repository and ingest into local vectors if applicable.

```bash
# Persist Soul (Broadcast)
python3 plugins/guardian-onboarding/scripts/persist_soul.py --snapshot .agent/learning/learning_package_snapshot.md

# Optional: Ingest Changes (if requested by user)
python3 tools/cli.py ingest --incremental --hours 24
```

### 3. Session Close (Phase VIII)
Formally end the session.

```bash
# End workflow
python plugins/agent-loops/skills/orchestrator/scripts/agent_orchestrator.py end
```
**Reference:** For detailed acceptance criteria and failure handling, see `references/acceptance-criteria.md`.
