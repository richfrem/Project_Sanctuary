---
name: guardian-onboarding
description: MANDATORY: Master initialization and closure skill for new agent sessions. Grounds the agent in Protocol 128 (Hardened Learning Loop), handles Orchestrator handoff, and executes the formal Seal and Persist closure sequences.
disable-model-invocation: false
---

# Guardian Onboarding (The Orientator)

**Version:** 5.0 (Protocol 128 Compliant)
**Purpose:** Ensure every agent session begins with a verified context download (Phase I) and routes to the `orchestrator` skill. Provide the explicit closure commands (Phase V-IX) to Seal and Persist the session.

## ⚡ Triggers (When to use this)
*   **Start of Session:** "I am a new agent session."
*   **End of Session:** "Seal the session," "Persist the soul," "Run the closure sequence."
*   **Protocol Check:** "What is the current learning protocol?"

## 🛡️ The Guardian Boot Sequence (Protocol 128)

Follow these steps **in order** to establish a valid session context.

### 1. The Anchor (Tactical Status)
**Goal:** Ingest current tactical directives and system status.
**Action:**
```
Read: .agent/learning/guardian_boot_digest.md
```
*   **Extract:** Active Tasks, System Status, and any "CRITICAL" alerts.
*   **Output:** "Guardian Status: [Status] | Active Directives: [Count]"

### 2. The Doctrine (Hardened Learning Loop)
**Goal:** Load the operational laws of the project (Protocol 128).
**Action:**
```
Read: plugins/guardian-onboarding/resources/protocols/128_Hardened_Learning_Loop.md
```
*   **Focus:** Phase checklist (I-VIII), Gate requirements (HITL), and Persistence obligations.
*   **Reference:** Architecture diagram: `plugins/guardian-onboarding/resources/protocols/protocol_128_learning_loop.mmd`

### 3. The Visual (Architecture)
**Goal:** Visualize the Orchestrator flow and closure sequence.
**Action:**
```
Read: plugins/guardian-onboarding/resources/protocols/protocol_128_learning_loop.mmd
```

### 4. Orchestrator Handoff
**Goal:** Transfer control to the `orchestrator` skill to determine the appropriate execution pattern (Learning, Red Team, Dual-Loop, Swarm).
**Action:** Let the user know you are ready, and invoke the orchestrator logic.

---

## 🔒 The Guardian Closure Sequence (Protocol 128 Phase V-VIII)

When a task or session is complete, the Guardian resumes control to orchestrate the unified closure sequence. The orchestrator retro occurs *first*, capturing learnings, before the Guardian formally seals the session state.

### Phase V: Orchestrator Retrospective
**Trigger:** Task execution complete.
**Action:** Let the Orchestrator generate the structured retrospective to capture learnings.
```bash
python3 plugins/agent-loops/skills/orchestrator/scripts/agent_orchestrator.py retro
```

### Phase VI: The Technical Seal
**Trigger:** Retrospective is complete and learnings are captured.
**Action:** Execute the seal script to generate the formal evidence bundle and update the learning package.
```bash
python3 plugins/guardian-onboarding/scripts/capture_snapshot.py --type seal
```
*If this fails (Iron Check / drift detected), you must halt and alert the user.*

### Phase VII: Soul Persistence
**Trigger:** Seal is successfully completed.
**Action:** Broadcast the verified state to Hugging Face.
```bash
python3 plugins/guardian-onboarding/scripts/persist_soul.py --snapshot .agent/learning/learning_package_snapshot.md

# Optional: Ingest Changes into local vector DB
python3 plugins/vector-db/skills/vector-db/scripts/ingest.py --incremental --hours 24
```

### Phase VIII: Session End
**Trigger:** Post-persistence.
**Action:** Formally end the loop.
```bash
python3 plugins/agent-loops/skills/orchestrator/scripts/agent_orchestrator.py end
```

---

## Plugin Dependencies

| Plugin/Skill | Phase | Role |
|:---|:---|:---|
| `plugins/guardian-onboarding/skills/session-bootloader/` | I | Boot sequence (debrief + Iron Check) |
| `plugins/guardian-onboarding/skills/session-closure/` | VI-VIII | Seal + Persist + Close |
| `plugins/guardian-onboarding/scripts/learning_debrief.py` | I | Generates Truth Anchor snapshot |
| `plugins/guardian-onboarding/scripts/guardian_wakeup.py` | I | Iron Check + boot digest |
| `plugins/guardian-onboarding/scripts/capture_snapshot.py` | VI | Seals context bundle via context-bundler |
| `plugins/guardian-onboarding/scripts/persist_soul.py` | VII | Uploads to HuggingFace |
| `plugins/guardian-onboarding/scripts/evolution_metrics.py` | — | Protocol 131: Fitness/Depth/Scope metrics |
| `plugins/guardian-onboarding/scripts/forge_llm.py` | — | Fine-tuned model queries via Ollama |
| `plugins/agent-loops/skills/orchestrator/` | II-V | Routes tasks, generates retrospective |
| `plugins/context-bundler/scripts/bundle.py` | VI | Called by capture_snapshot to produce bundle |
| `plugins/env-helper/scripts/env_helper.py` | VII | Resolves HF_TOKEN / HF_USERNAME |
| `plugins/rlm-factory/` | I, VI | RLM cache read (boot) and synthesis (seal) |
