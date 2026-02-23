---
name: guardian-onboarding
description: MANDATORY: Master initialization and closure skill for new agent sessions. Grounds the agent in Protocol 128 (Hardened Learning Loop), handles Orchestrator handoff, and executes the formal Seal and Persist closure sequences.
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
Read: .agent/workflows/sanctuary_protocols/sanctuary-learning-loop.md
```
*   **Focus:** Phase checklist (I-X), Gate requirements (HITL), and Persistence obligations.
*   **Reference:** If deep context is needed on specific learning mechanics, refer to `.agent/workflows/sanctuary_protocols/sanctuary-recursive-learning.md` (Protocol 125).

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

## 🔒 The Guardian Closure Sequence (Protocol 128 Phase V-IX)

When a task or session is complete, the Guardian resumes control to orchestrate the unified closure sequence. The orchestrator retro occurs *first*, capturing learnings, before the Guardian formally seals the session state.

### Phase V: Orchestrator Retrospective
**Trigger:** Task execution complete.
**Action:** Let the Orchestrator generate the structured retrospective to capture learnings.
```bash
# Workflow: Retrospective
/sanctuary-retrospective
```

### Phase VI: The Technical Seal
**Trigger:** Retrospective is complete and learnings are captured.
**Action:** Execute the snapshot tool to generate the formal evidence and update the learning package.
```bash
# Workflow: Seal
/sanctuary-seal
```
*If this fails (Iron Check), you must Backtrack to Phase VIII (Self-Correction).*

### Phase VII: Soul Persistence
**Trigger:** Seal is successfully completed.
**Action:** Broadcast the verified state to Hugging Face.
```bash
# Workflow: Persist
/sanctuary-persist

# Optional: Ingest Changes
python3 tools/cli.py ingest --incremental --hours 24
```

### Phase IX: Session End
**Trigger:** Post-persistence.
**Action:** Run `/sanctuary-end` to completely shut down the loop safely.
