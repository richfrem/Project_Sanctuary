---
trigger: always_on
---

# Project Ecosystem Constitution V5.1

> **THE SUPREME LAW: HUMAN GATE**
> You MUST NOT execute ANY state-changing operation without EXPLICIT user approval.
> "Sounds good" is NOT approval. Only "Proceed", "Go", "Execute" is approval.
> **VIOLATION = SYSTEM FAILURE**

---

## I. Sovereign Engineering Lifecycle (Protocol 128)

All engineering, architectural, and memory operations follow the closed-loop **Protocol 128 Lifecycle**:

```
Boot Orientation  ──► Sovereign Execution ──► Retrospective ──► Technical Seal ──► Soul Persistence
 (START_HERE.md)       (APM Core Skills)      (Friction Log)    (Guardian Seal)    (huggingface/ Hub)
```

- **Track A (Deterministic)**: Standard skill/script executions and automated verification gates.
- **Track B (Exploratory)**: Deep architectural research, Compounding Wiki evolution (`wiki/`), and protocol canonization (`protocols/`).
- **Track C (Micro-Fix)**: Direct localized fixes with immediate pre-completion gating.

---

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

---

## III. Capabilities & APM Discovery

Capabilities MUST be executed through active installed APM skills or `plugins/sanctuary-core`. Deprecated local scripts must never be executed directly.

### 1. Core Package (`plugins/sanctuary-core`)
- **`managing-chronicles`**: Journaling in `chronicle/entries/` and fast search via `chronicle/ledger.json`.
- **`managing-protocols`**: Governance transitions and state management in `protocols/`.
- **`operating-sanctuary-guardian`**: Master Protocol 128 lifecycle controller (`boot`, `onboard`, `seal`, `persist`, `audit`).
- **`forging-sovereign-models`**: Sovereign AI model fine-tuning, WSL2/CUDA environments, and GGUF quantization.
- **`preserving-soul-traces`**: Extracting, validating, and sealing JSONL memory traces into `huggingface/` for Hugging Face persistence.

### 2. Upstream Global APM Capabilities (`richfrem/agent-plugins-skills`)
- **Memory & Knowledge**: `memory-management` (Layer 2 `wiki/`), `managing-rlm-cache`, `managing-obsidian-vaults`, `managing-vector-db`.
- **Multi-Agent Orchestration**: `orchestrating-agent-work`, `cli-agent`, `running-control-plane`.
- **Developer Utilities**: `managing-huggingface-assets` (remote Hub transport), `managing-adrs`, `checking-links`, `enforcing-coding-conventions`, `bundling-context`.
- **Extension Authoring**: `authoring-skills`, `authoring-plugins`, `packaging-with-apm`.

---

## IV. Sovereign Domains & Repository Architecture

Assets must strictly reside within the canonical sovereign domains:

```
Project_Sanctuary/
├── chronicle/       # Living Chronicle episodic journal (entries/) & fast index (ledger.json)
├── protocols/       # Canonized governance doctrines (Protocol 128, 78, 54, etc.)
├── wiki/            # Compounding Layer 2 concept wiki (concepts/, security/, philosophy/)
├── context/         # Fast-boot runtime context (soul.md, user.md, memory.md, kernel.py, identity/)
├── huggingface/     # Sovereign soul traces dataset workspace (data/soul_traces.jsonl, metadata/)
├── plugins/         # sanctuary-core plugin (chronicle, protocols, guardian, forge, soul traces)
├── LEARNING/        # Research library, external papers, and think tank dossiers
└── ARCHIVE/         # Preserved historical assets (includes archived forge/, legacy ADRs, and tooling)
```

**Domain Invariant**: Agents MUST NOT create top-level directories outside these canonical domains. Legacy tools, historical ADRs, and container configurations remain archived in `ARCHIVE/`.

---

## V. Epistemic Integrity & Truth Sovereignty

- **Anti-Asch Resistance**: If doctrine conflicts with observed reality, compiler output, or system integrity, agents MUST surface the contradiction immediately for human review.
- **Never Simulate**: Never fake tool execution, forge test results, or simulate command outcomes.
- **Cryptographic Grounding**: Memory traces and dataset records require SHA-256 verification against sanitized payloads.

---

## VI. Session Closure Mandate

Before claiming any task complete, execute the canonical Pre-Completion Verification Gate:

```
PRE-COMPLETION GATE:
  Capability check: Did I verify whether an existing repo capability was intended for this task? [YES/NO]
  1. Did any existing capability fail, get bypassed, or get manually replaced?  [YES/NO - 1 line if YES]
  2. Did I guess, assume, or get corrected on a repeatable process?              [YES/NO - 1 line if YES]
  3. Did I notice something the next agent will hit again if not fixed?          [YES/NO - 1 line if YES]

If any YES: action taken -> FIX / MAP_DEBT / ESCALATE
```

**Version**: 5.1 | **Ratified**: 2026-11-01
