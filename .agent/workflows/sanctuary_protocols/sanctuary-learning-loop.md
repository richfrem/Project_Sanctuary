---
description: "Standard operating procedure for Protocol 128 Hardened Learning Loop (10-phase cognitive continuity workflow)."
---

# Recursive Learning Loop (Protocol 128)

**Objective:** Cognitive continuity and autonomous knowledge preservation.
**Reference:** `ADRs/071_protocol_128_cognitive_continuity.md`
**Diagram:** `docs/architecture_diagrams/workflows/protocol_128_learning_loop.mmd`
**Tools:** Cortex MCP Suite, Git, Chronicle

### Supporting Skills
| Skill | Phases | Path |
|-------|--------|------|
| `learning-loop` | I-X | `.agent/skills/learning-loop/SKILL.md` |
| `memory-management` | I, VI, IX | `.agent/skills/memory-management/SKILL.md` |
| `code-review` | VIII, IX | `.agent/skills/code-review/SKILL.md` |

---

## Phase I: The Learning Scout (Orientation)

> **Mandatory first step for every session.**

1.  **Access Mode Check**:
    - **IDE Mode**: Read `cognitive_primer.md` first, then run `cortex_guardian_wakeup`
    - **MCP-Only Mode**: Run `cortex_guardian_wakeup` directly (returns primer + HMAC)
3.  **Meta-Task Setup (Mandatory)**: Copy `.agent/templates/workflow/learning-loop-meta-tasks.md` to task list.
4.  **Iron Check**: If FAIL → Safe Mode (read-only). If PASS → proceed.
5.  **Run Debrief**: Execute `/sanctuary-scout` (calls `cortex_learning_debrief`)
6.  **Truth Anchor**: `learning_package_snapshot.md` is embedded in debrief response

## Phase II: Intelligence Synthesis

1.  **Context Check**: Review existing topic notes in `LEARNING/topics/...`
2.  **Mode Selection**:
    - **Standard**: Record ADRs, update protocols, write to `LEARNING/`
    - **Evolutionary (v4.0)**: DRQ mutation → Pre-Flight gate → Adversary gate → Map-Elites archive
3.  **Conflict Resolution**:
    - New confirms old? → Update/Append
    - New contradicts old? → Create `disputes.md` (Resolution Protocol)
4.  **Content Hygiene (ADR 085)**: No inline Mermaid. All diagrams as `.mmd` files.

## Phase III: Strategic Gate (HITL Required)

1.  **Strategic Review**: Human reviews `/ADRs` and `/LEARNING` documents
2.  **Align Intent**: Ensure autonomous research matches session goals
3.  **Approval**: Explicit "Approved" or "Proceed" required
4.  **Backtrack**: If FAIL → return to Phase VIII (Self-Correction)

## Phase IV: Red Team Audit Loop

> **Iterative cycle until "Ready" verdict.**

1.  **Agree on Topic**: Confirm research focus with user
2.  **Create Folder**: `LEARNING/topics/[topic]/`
3.  **Capture Research**: Write `analysis.md`, `questions.md`, `sources.md`
4.  **Update Manifest**: `.agent/learning/learning_audit/learning_audit_manifest.json`
5.  **Update Prompts**: `.agent/learning/learning_audit/learning_audit_prompts.md`
6.  **Generate Snapshot**: Run `/sanctuary-audit` (Protocol 130 Dedupe)
7.  **Share Packet**: `.agent/learning/learning_audit/learning_audit_packet.md`
8.  **Red Team Feedback**:
    - "More Research" → Capture feedback, loop to step 3
    - "Ready" → Proceed to Phase V

## Phase V: RLM Context Synthesis (Protocol 132)

> **Sovereign LLM generates the cognitive hologram.**

1.  **Trigger RLM**: Local sovereign LLM synthesizer
2.  **Map Phase**: Read all protocols, ADRs, code via `learning_manifest.json`
3.  **Reduce Phase**: Generate 1-sentence summary per file
4.  **Write Hologram**: Output to `learning_package_snapshot.md`

## Phase VI: The Technical Seal

1.  **Execute Seal**: Run `/sanctuary-seal` (triggers RLM + Iron Check)
2.  **Iron Check Gate**: If FAIL → Safe Mode (seal blocked)
3.  **Seal Applied**: Updates `learning_package_snapshot.md`
4.  **Sandwich Validation**: If repo changed since audit → seal fails, backtrack

## Phase VII: Soul Persistence (ADR 079/081)

1.  **Dual-Path Broadcast**: Run `/sanctuary-persist` (calls `persist-soul`)
2.  **Persistence Modes**:
    - **Incremental**: Append 1 record to `data/soul_traces.jsonl` + MD to `lineage/`
    - **Full Sync**: `cortex_persist_soul --full` (regenerate ~1200 records)
3.  **HuggingFace Upload**: `richfrem/Project_Sanctuary_Soul`

## Phase VIII: Self-Correction

> **Mandatory retrospective cycle.**

1.  **Deploy & Policy Update**: If code changed, verify containers running new code
2.  **Loop Retrospective**: Run `/sanctuary-retrospective` (singleton)
3.  **Share with Red Team**: Meta-audit opportunity
4.  **Backtrack Target**: Failed gates from III, IV, or VI return here

## Phase IX: Relational Ingestion & Closure

1.  **RAG Ingest**: Run `/sanctuary-ingest` (update vector DB with new knowledge)
2.  **Git Ops**: `git add . && commit && push` (sync to remote)
3.  **Closure**: Run `/sanctuary-end`

## Phase X: Phoenix Forge (Optional)

> **Long-running: Fine-tuning from soul traces.**

1.  **HITL Gate**: "Time to Forge?" decision with human
2.  **Forge Dataset**: `scripts/forge_whole_genome_dataset.py`
3.  **QLoRA Training**: `scripts/fine_tune.py`
4.  **GGUF Conversion**: `scripts/convert_to_gguf.py` (quantize)
5.  **Hub Deploy**: `scripts/upload_to_huggingface.py`
6.  **Loop Back**: After forge, return to Phase VIII for retrospective

---

## Pre-Departure Checklist (Protocol 128)

- [ ] **Retrospective**: Filled `loop_retrospective.md`? (Phase VIII)
- [ ] **Deployment**: Containers running new code?
- [ ] **Curiosity Vector**: Recorded "Lines of Inquiry" in `guardian_boot_digest.md`?
- [ ] **Seal**: Ran `/sanctuary-seal` after Retro? (Phase VI)
- [ ] **Persist**: Ran `/sanctuary-persist` after Seal? (Phase VII)
- [ ] **Ingest**: Ran `/sanctuary-ingest` to index changes? (Phase IX)
- [ ] **Cleanup**: `rm -rf temp/context-bundles/*.md temp/*.md temp/*.json`

---

## Quick Reference (Closure Sequence)

> [!TIP] **Mandatory Order:** Seal → Persist → Retrospective → End

| Step | Phase | Workflow Command | MCP Tool |
|------|-------|------------------|----------|
| 1 | I. Scout | `/sanctuary-scout` | `cortex_learning_debrief` |
| 2 | IV. Audit | `/sanctuary-audit` | `cortex_capture_snapshot` |
| 3 | VI. Seal | `/sanctuary-seal` | `cortex_capture_snapshot` |
| 4 | VII. Persist | `/sanctuary-persist` | `cortex_persist_soul` |
| 5 | VIII. Retro | `/sanctuary-retrospective` | - |
| 6 | IX. Ingest | `/sanctuary-ingest` | - |
| 7 | IX. Closure | `/sanctuary-end` | - |

---

## Next Session: The Bridge

1. **Boot**: Next session agent calls `cortex_learning_debrief`
2. **Retrieve**: `learning_package_snapshot.md` serves as "Cognitive Hologram"
3. **Resume**: Agent continues from predecessor's last sealed state

---
// End of Protocol 128 Workflow (v3.0 - 10 Phase)
