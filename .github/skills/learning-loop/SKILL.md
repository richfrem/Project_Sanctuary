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

## The Iron Chain

```
Scout → Synthesize → Gate → Audit → Seal → Persist → Retrospective → Ingest → End
```

---

## Session Open (MANDATORY)

### Step 1: Read boot files
```
.agent/learning/guardian_boot_contract.md    → Constraints
.agent/learning/cognitive_primer.md          → Role
.agent/learning/learning_package_snapshot.md → Context (read first 500 lines)
```
**PROOF**: You must quote at least one line from each file to confirm you read them.

### Step 2: Run Scout
```bash
/sanctuary-scout
```
**PROOF**: Paste the scout output. If it fails, report the error — do NOT fabricate success.

### Step 3: Report readiness
Tell the user: "Boot complete. [summary of context loaded]. Ready to work."

**STOP**: Do NOT proceed to work until you have completed all 3 steps above.

---

## Session Close (MANDATORY — DO NOT SKIP)

> **This sequence is non-negotiable.** If the user says "done", "wrap up",
> "end session", or similar — you MUST run these steps IN ORDER.
> Skipping any step means the next session starts blind.

### Step 1: Seal (Phase VI)
```bash
/sanctuary-seal
```
**PROOF**: Paste the seal output showing snapshot was created.
**STOP**: Do NOT proceed to Persist until seal output is confirmed.

### Step 2: Persist (Phase VII)
```bash
/sanctuary-persist
```
**PROOF**: Paste the persist output showing HuggingFace upload status.
**STOP**: Do NOT proceed to Retrospective until persist output is confirmed.

### Step 3: Retrospective (Phase VIII)
```bash
/sanctuary-retrospective
```
**PROOF**: Paste the retrospective output. This is a self-audit — actually reflect on what happened.

### Step 4: End (Phase IX)
```bash
/sanctuary-end
```
**PROOF**: Paste the end output showing git status and closure confirmation.

**DONE**: Only after all 4 steps have real output can you consider the session closed.

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
| V (Verify) | Outer Loop inspects Inner Loop output | `verify_inner_loop_result.py` |
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
