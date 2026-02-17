---
name: learning-loop
description: "Protocol 128 Hardened Learning Loop for cognitive continuity across agent sessions. Use when: (1) starting any session (boot/orientation), (2) performing research or knowledge capture, (3) closing a session (seal, persist, retrospective), (4) user invokes /sanctuary-start, /sanctuary-learning-loop, /sanctuary-scout, /sanctuary-audit, /sanctuary-seal, /sanctuary-persist, /sanctuary-retrospective, /sanctuary-ingest, /sanctuary-end, or any Protocol 128 phase (I-X). Ensures knowledge is captured, validated, and persisted so the next session can resume from a sealed cognitive state."
---

# Learning Loop (Protocol 128)

Structured 10-phase cognitive continuity loop ensuring knowledge survives across agent sessions.

## CRITICAL: Anti-Simulation Rules

> **YOU MUST ACTUALLY RUN EVERY COMMAND LISTED BELOW.**
> Describing what you "would do", summarizing expected output, or marking
> a step complete without pasting real tool output is a **PROTOCOL VIOLATION**.
>
> **Proof = pasted command output.** No output = not done.
>
> **Closure is NOT optional.** If the user says "end session" or you are
> wrapping up, you MUST run the full closure sequence (Seal → Persist →
> Retrospective → End). Skipping any step is a failure.

### Known Agent Failure Modes (DO NOT DO THESE)
1. **Checkbox theater**: Marking `[x]` without running the command
2. **Output fabrication**: Writing "Seal complete" without running `/sanctuary-seal`
3. **Closure skip**: Ending a session without seal/persist/retrospective
4. **Role-playing**: Narrating "I would now run..." instead of actually running it
5. **Rushing**: Skipping steps to "save time" — every step exists for a reason

---

---

## The Iron Chain

> **Prerequisite**: You must have established a Valid Session via `/sanctuary-start` or a manual boot sequence (Protocol 128 Phase I).

```
Scout → Synthesize → Gate → Audit → [Execution / Dual-Loop] → Seal → Persist → Retrospective → Ingest → End
```

---

### Phase I: The Learning Scout (Orientation)

> **Goal**: Establish Identity & Context.
> **Trigger**: `/sanctuary-start` (Automated) OR Manual Boot.

1.  **Identity Check**:
    *   Read `.agent/learning/cognitive_primer.md` (Role & Directives)
    *   Read `.agent/learning/guardian_boot_contract.md` (Constraints)
    
2.  **Context Loading (Wakeup)**:
    *   **Command**: `/sanctuary-scout`
    *   **Action**: Calls `cortex_learning_debrief` to load the "Cognitive Hologram".
    *   **Verify**: Did you receive the `learning_package_snapshot.md` content?

3.  **Iron Check**:
    *   Confirm integrity. If FAIL -> Stop (Safe Mode).

4.  **Report Readiness**:
    *   "Boot complete. Context loaded. Ready."

**STOP**: Do NOT proceed to work until you have completed Phase I.

**STOP**: Do NOT proceed to work until you have completed all 3 steps above.

---

## The Bridge (Why we Seal)

> **Cognitive Continuity**: We Seal (Phase VI) and Persist (Phase VII) so that the **Next Agent** can Resume (Phase I).
> *   **Snapshot**: Your `seal` output becomes their `boot` input.
> *   **Trace**: Your `persist` output teaches the Soul model.

---

## Supporting Skills (The Toolkit)

| Skill | Phase | Purpose |
|-------|-------|---------|
| **Spec Kitty** | II, III, V | Architecture & Planning (`/spec-kitty.*`) |
| **Dual Loop** | V | Execution Delegation (`dual-loop-supervisor`) |
| **Context Bundling** | II, IV | Context Sharing & Review Packets |
| **Memory Mgmt** | I, VI, VII | Cognitive Continuity & Persistence |

---

## Work Phase (Phases II-V)

### Phase II: Intelligence Synthesis
1.  **Mode Selection**:
    *   **Standard**: Update docs/ADRs.
    *   **Evolutionary (v4.0)**: DRQ Mutation -> Adversary Gate -> Map-Elites (Advanced).
2.  **Synthesis**: Do your research.

### Phase III: Strategic Gate (HITL)
> **Human-in-the-Loop Required**
1.  **Review**: Present findings to User.
2.  **Gate**: Wait for explicit "Approved" or "Proceed".
    *   *If FAIL*: Backtrack to Phase VIII (Self-Correction).

### Phase IV: Audit (Red Team)
1.  **Command**: `/sanctuary-audit`
2.  **Action**: User reviews the plan/architecture.

### Execution Branch (Post-Audit)
> **Choose your Execution Mode:**

**Option A: Standard Agent (Single Loop)**
*   **Action**: You write the code, run tests, and verify yourself.
*   **Protocol**: Stay in this session.

**Option B: Dual Loop (Protocol 133)**
*   **Action**: Delegate execution to Inner Loop.
*   **Command**: Open `dual-loop-supervisor` SKILL. Execute starting from **Step 1**.
*   **Return**: Once Inner Loop finishes (Step 5 "Done"), resume here at **Phase V (RLM Synthesis)**.

---

### Phase X: Phoenix Forge (Optional)
> **Long-running Fine-Tuning**
1.  **Trigger**: User request or "Time to Forge" decision.
2.  **Action**: Forge dataset -> QLoRA Train -> Deploy new model.

---

## Session Close (MANDATORY — DO NOT SKIP)

> **This sequence is non-negotiable.** If the user says "done", "wrap up",
> "end session", or similar — you MUST run these steps IN ORDER.
> Skipping any step means the next session starts blind.

### Phase V: RLM Synthesis (Protocol 132)
> **Automated by Seal**. The sovereign LLM summarizes your work.
> *Ensure `ollama-launch` has been run.*

### Phase VI: The Technical Seal
1.  **Command**: `/sanctuary-seal`
2.  **Action**: Snapshots state, runs Iron Check, creates `snapshot.md`.
3.  **Gate**: Must PASS to proceed.

### Phase VII: Soul Persistence
1.  **Command**: `/sanctuary-persist`
2.  **Action**: Uploads session traces to HuggingFace (Long-term Memory).

### Phase VIII: Self-Correction (Retrospective)
1.  **Command**: `/sanctuary-retrospective`
2.  **Action**: Analyze what went well/poorly.

### Phase IX: Ingest & End
1.  **Command**: `/sanctuary-ingest` (Update Vector DB)
2.  **Command**: `/sanctuary-end` (Git Commit & Close)

---

## Phase Reference

| Phase | Name | Command | Proof Required |
|-------|------|---------|----------------|
| I | Scout | `/sanctuary-scout` | Paste output |
| II | Synthesis | *(manual)* | Show files created/modified |
| III | Strategic Gate | *(HITL)* | User says "Approved"/"Proceed" |
| IV | Red Team Audit | `/sanctuary-audit` | Paste output |
| V | RLM Synthesis | *(sovereign LLM)* | Show hologram update |
| VI | Seal | `/sanctuary-seal` | Paste output |
| VII | Persist | `/sanctuary-persist` | Paste output |
| VIII | Self-Correction | `/sanctuary-retrospective` | Paste output |
| IX | Ingest & Close | `/sanctuary-ingest` → `/sanctuary-end` | Paste both outputs |
| X | Phoenix Forge | *(optional)* | N/A |

---

## Task Tracking Rules

> **You are not "done" until the kanban says you're done.**

- Use `.kittify/scripts/tasks/tasks_cli.py` to move tasks between lanes
- **NEVER** manually edit `[x]` checkboxes in tasks.md — use the CLI
- **NEVER** mark a task `done` without running its verification command first
- After moving a task, run `/spec-kitty.status` and paste the board to confirm the move registered

---

## Dual-Loop Integration (Protocol 133)

When Protocol 128 runs inside a Dual-Loop session:

| Phase | Dual-Loop Role | Notes |
|-------|---------------|-------|
| I (Scout) | Outer Loop boots, orients | Reads boot files + spec context |
| II-III (Synthesis/Gate) | Outer Loop plans, user approves | Strategy Packet generated |
| IV (Audit) | Outer Loop snapshots before delegation | Pre-execution checkpoint |
| *(Execution)* | **Inner Loop** performs tactical work | Code-only, no git |
| *Verification* | Outer Loop inspects Inner Loop output | `verify_inner_loop_result.py` |
| V (RLM Synthesis) | Outer Loop (Automated) | Cognitive Hologram generation |
| VI-IX (Seal→End) | Outer Loop closure | Standard seal/persist/retro/end |

**Key rule**: The Inner Loop does NOT run Learning Loop phases. All cognitive continuity is the Outer Loop's responsibility.

**Cross-reference**: [dual-loop-supervisor SKILL](../dual-loop-supervisor/SKILL.md) | [Protocol 133 workflow](../../workflows/sanctuary_protocols/dual-loop-learning.md)

---

## Infrastructure Prerequisites

Before running closure phases, check these services:

| Service | Needed For | Check Command | Start Skill |
|---------|-----------|---------------|-------------|
| **Ollama** | Phase VI (Seal) | `curl -sf http://127.0.0.1:11434/api/tags` | `ollama-launch` |
| **ChromaDB** | Phase IX (Ingest) | `curl -sf http://localhost:8110/api/v2/heartbeat` | `vector-db-launch` |

Run the check commands BEFORE starting closure. If either fails, run the corresponding skill first.

---

## Key Files

| File | Purpose |
|------|---------|
| `.agent/learning/cognitive_primer.md` | Role orientation |
| `.agent/learning/learning_package_snapshot.md` | Cognitive Hologram |
| `.agent/learning/guardian_boot_contract.md` | Immutable constraints |
| `.agent/learning/guardian_boot_digest.md` | Tactical status |
| `ADRs/071_protocol_128_cognitive_continuity.md` | Protocol ADR |
| `docs/architecture_diagrams/workflows/protocol_128_learning_loop.mmd` | Flow diagram |
