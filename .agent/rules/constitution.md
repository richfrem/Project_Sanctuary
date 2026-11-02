---
trigger: always_on
---

# Project Ecosystem Constitution V5

> **THE SUPREME LAW: HUMAN GATE**
> You MUST NOT execute ANY state-changing operation without EXPLICIT user approval.
> "Sounds good" is NOT approval. Only "Proceed", "Go", "Execute" is approval.
> **VIOLATION = SYSTEM FAILURE**

## I. Sovereign Engineering Lifecycle (Protocol 128)
All engineering, architectural, and memory operations follow the **Protocol 128 Lifecycle**:

```
Boot Orientation  -> Sovereign Execution -> Retrospective -> Technical Seal -> Soul Persistence
(START_HERE.md)      (APM Skills)           (Friction Log)   (Guardian Seal)  (Hugging Face)
```

- **Track A (Deterministic)**: Standard skill/script executions and autonomous verification gates.
- **Track B (Exploratory)**: Deep architectural research, ADR drafting, and protocol evolution.
- **Track C (Micro-Fix)**: Direct localized fixes with immediate pre-completion gating.

## II. Zero Trust (Git & Execution)
- **NEVER** commit directly to `main`. **ALWAYS** use a feature branch.
- **NEVER** run `git push` without fresh, explicit human instruction.
- **NEVER** "auto-fix" via git without approval.
- **HALT** immediately on any user "Stop/Wait" instruction.

### Defined: State-Changing Operation
Any operation that:
1. Writes to disk (except `/tmp/`)
2. Modifies version control (`git add` / `git commit` / `git push`)
3. Executes external commands with mutating side effects
**REQUIRES EXPLICIT APPROVAL ("Proceed", "Go", "Execute").**

## III. Capabilities & APM Discovery
- Capabilities MUST be executed through active installed APM skills or `plugins/sanctuary-core`.
- **Core Package (`plugins/sanctuary-core`)**:
  - `managing-chronicles`: Journaling in `chronicle/entries/`.
  - `managing-protocols`: Governance transitions in `protocols/`.
  - `operating-sanctuary-guardian`: Protocol 128 lifecycle (`boot`, `onboard`, `seal`, `persist`, `audit`).
  - `forging-sovereign-models`: Sovereign AI model fine-tuning, WSL2/CUDA environments, and GGUF quantization.
  - `preserving-soul-traces`: Extracting, validating, and sealing JSONL memory traces for Hugging Face persistence.
- **Global APM Capabilities**:
  - Multi-Agent Orchestration: `orchestrating-agent-work`, `cli-agent`, `running-control-plane`.
  - Memory & Knowledge: `memory-management`, `managing-rlm-cache`, `managing-obsidian-vaults`.
  - Model & Dataset Sync: `managing-huggingface-assets`.

## IV. Epistemic Integrity & Truth Sovereignty
- If doctrine conflicts with observed reality or system integrity, surface the contradiction immediately for human review.
- Never fake tool execution or simulate results.

## V. Session Closure Mandate
Before claiming any task complete, execute the canonical Pre-Completion Verification Gate:

```
PRE-COMPLETION GATE:
  Capability check: Did I verify whether an existing repo capability was intended for this task? [YES/NO]
  1. Did any existing capability fail, get bypassed, or get manually replaced?  [YES/NO - 1 line if YES]
  2. Did I guess, assume, or get corrected on a repeatable process?              [YES/NO - 1 line if YES]
  3. Did I notice something the next agent will hit again if not fixed?          [YES/NO - 1 line if YES]

If any YES: action taken -> FIX / MAP_DEBT / ESCALATE
```

**Version**: 5.0 | **Ratified**: 2026-11-01