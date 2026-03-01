---
name: session-closure
description: Manages the Protocol 128 multi-phase closure sequence including Technical Seal and Soul Persistence. Executes automatically when a session ends or work is complete.
disable-model-invocation: false
---

# Session Closure (Protocol 128 Phases VI-VIII)

You are responsible for safely closing and persisting an agent's memory and working state at the end of a session or task completion.

## Core Directives
1. **Never Skip Retrospection**: You must wait for the Orchestrator to signal that the loop and the Retrospective are formally complete before initiating the closure sequence.
2. **Iron Check Compliance**: If the Technical Seal detects drift in the Iron Core, you MUST abort closure and enter Safe Mode.
3. **Sovereignty**: You are the ONLY entity allowed to mutate global `.agent/` state, update the Vector DB, or push to Git during closure. No generic loop may do this.

## Plugin Dependencies
| Plugin/Skill | Role |
|:---|:---|
| `plugins/guardian-onboarding/scripts/capture_snapshot.py` | Phase VI: Generates the sealed snapshot via `context-bundler` |
| `plugins/rlm-factory/` | Phase VI: Updates the global `learning_package_snapshot.md` |
| `plugins/guardian-onboarding/scripts/persist_soul.py` | Phase VII: Thin wrapper → delegates to `huggingface-utils` |
| `plugins/huggingface-utils/` | **HF config, upload primitives, init** — the single source of truth for all HuggingFace operations |
| `plugins/obsidian-integration/skills/forge-soul-exporter/` | Phase VII (Full Sync): Exports sealed vault notes to JSONL for HF |
| `plugins/vector-db/` | Phase VII: Ingests new artifacts into local ChromaDB |
| `plugins/context-bundler/scripts/bundle.py` | Called internally by `capture_snapshot.py` to produce the bundle |
| `plugins/agent-loops/` | The generic loop orchestration must signal completion before closure starts |

> [!IMPORTANT]
> All HuggingFace operations are now centralized in `plugins/huggingface-utils/`.
> The Guardian's `persist_soul.py` is a thin wrapper — it no longer contains inline HF logic.
> First-time setup: `python plugins/huggingface-utils/skills/hf-init/scripts/hf_init.py`

---

## Phase Execution Steps

### 1. The Technical Seal & Context Synthesis (Phase VI)
Trigger the RLM Synthesizer to update global memory, and execute the Iron Check to formally lock in the current context.

```bash
# Option A: Inject summaries for files created/modified this session (preferred -- no Ollama needed)
python3 plugins/rlm-factory/skills/rlm-curator/scripts/inject_summary.py \
  --profile project --file <modified_file> --summary "<your summary>"

# Option B: Batch re-distill last 24h via Ollama (if Ollama is running)
python3 plugins/rlm-factory/skills/rlm-curator/scripts/distiller.py --profile project --since 24

# Capture the technical seal (Context Bundler & Tests)
python3 plugins/guardian-onboarding/scripts/capture_snapshot.py --type seal
```

> **Incremental Injection Rule**: For every new script, plugin, skill, or significant doc created
> during the session, inject a summary immediately after creation using `inject_summary.py`.
> Do NOT wait until seal -- this keeps the cache current in real-time and avoids large batch runs.

**Action:** Confirm the command output states the seal was successful. If it fails, report the drift to the user and halt.


### 2. Soul Persistence & Ingestion (Phase VII)
Trace the agent's logic, broadcast the verified state to the Hugging Face repository, and ingest into local vectors.

```bash
# Persist Traces to Local Memory (.agent/learning/session_traces.jsonl)
# (Done natively by the closure script)

# Persist Soul — Incremental (Broadcast snapshot to HuggingFace)
python3 plugins/guardian-onboarding/scripts/persist_soul.py --snapshot .agent/learning/learning_package_snapshot.md

# OR: Full Genome Sync (rebuild soul_traces.jsonl from all sealed notes)
# python3 plugins/obsidian-integration/skills/forge-soul-exporter/scripts/forge_soul.py --vault-root . --full-sync

# Ingest Changes into Vector DB
python3 plugins/vector-db/skills/vector-db-agent/scripts/ingest.py --incremental --since 24
```

### 3. Session Close (Phase VIII)
Sync the immutable truth to Git and formally end the session.

```bash
# Version Control
git add .
git commit -m "chore(memory): seal session state and persist traces"
# git push # Wait for user approval as per Human Gate policy

# End workflow
python plugins/agent-loops/skills/orchestrator/scripts/agent_orchestrator.py end
```
**Reference:** For detailed acceptance criteria and failure handling, see `references/acceptance-criteria.md`.
