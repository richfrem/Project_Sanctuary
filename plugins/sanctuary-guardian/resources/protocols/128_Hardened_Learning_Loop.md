# Protocol 128: Orchestrated Cognitive Continuity & Execution Protocols

## 1. Objective
Establish a persistent, tamper-proof, and high-fidelity mechanism for orchestrating autonomous execution loops and capturing validated cognitive state deltas between agent sessions. This protocol routes workflows into explicit execution patterns (Learning, Red Team, Swarm, Dual-Loop) while enforcing a mandatory unified closure sequence.

## 2. The Red Team Gate (Zero-Trust Mode)
No cognitive update may be persisted to the long-term Cortex without meeting the following criteria:
8. In section 2 (Red Team Gate), the `cortex_learning_debrief` plugin skill reference should now be:
   - **Script**: `python3 plugins/sanctuary-guardian/scripts/learning_debrief.py --hours 24`
2. **Discrepancy Reporting**: The tool must highlight any gap between the agent's internal claims and the statistical reality on disk.
3. **HITL Review**: A human steward must review the targeted "Red Team Packet" (Briefing, Manifest, Snapshot) before approval.

## 3. The Integrity Wakeup (Bootloader)
Every agent session must initialize via the Protocol 128 Bootloader:
1. **Semantic HMAC Check**: Validate the integrity of critical caches using whitespace-insensitive JSON canonicalization.
2. **Debrief Ingestion**: Automatically surface the most recent verified debrief into the active context.
3. **Cognitive Primer**: Mandate alignment with the project's core directives before tool execution.

## 3A. The Iron Core & Safe Mode Protocol (Zero-Drift)
To prevent "identity drift" (Source: Titans [16]), we enforce a set of immutable files (Iron Core) that define the agent's fundamental nature.

### The Iron Check
A cryptographic verification runs at **Boot** (Guardian) and **Snapshot** (Seal). It validates that the following paths have not been tampered with:
- `01_PROTOCOLS/*`
- `ADRs/*`
- `founder_seed.json`
- `cognitive_continuity_policy.md`

### Safe Mode State Machine
If an Iron Check fails, the system enters `SAFE_MODE`.
- **Trigger**: Any uncommitted change to Iron Core paths without "Constitutional Amendment" (HITL Override).
- **Restrictions**: 
  - `EXECUTION` capability revoked (Read-only tools only).
  - `persist-soul` blocked.
  - `snapshot --seal` blocked.
- **Recovery**: Manual revert of changes or explicit `--override-iron-core` flag.

## 4. Technical Architecture (The Mechanism)

The system has evolved from a monolithic loop into a **Tri-Track Spec-Kitty** framework feeding into an **Orchestration** structure.

### A. Strategic Framing (Spec-Kitty Tracks)
Before task execution, work is triaged and framed:
*   **Track A**: Factory Standardized Workflows (`/codify-*`)
*   **Track B**: Discovery Features (`/spec-kitty.specify` → `plan` → `tasks`)
*   **Track C**: Micro-tasks (Trivial maintenance)

### B. Execution Routing (Orchestrator)
Once the Work Packages are defined by Spec-Kitty, tasks are routed by the `orchestrator` skill (`plugins/agent-loops/skills/orchestrator/`) into one of four execution patterns:

#### Pattern 1: Simple Learning Loop
- **Purpose**: Self-directed research, documentation, and knowledge synthesis.
- **Skill**: `plugins/agent-loops/skills/learning-loop/`
- **Flow**: Research & Synthesize -> Document Findings -> Iterate -> Seal

### B. Pattern 2: Red Team Review Loop
- **Purpose**: Audits, architectures, and design decisions requiring adversarial scrutiny.
- **Skill**: `plugins/agent-loops/skills/red-team-review/`
- **Flow**: Capture Research -> Bundle Context (`context-bundler`) -> Red Team Feedback -> Gate 2 (HITL) -> Seal

### C. Pattern 3: Dual-Loop Delegation
- **Purpose**: Execution of single, well-defined work packages.
- **Skill**: `plugins/agent-loops/skills/dual-loop/`
- **Flow**: Plan & Partition (Outer Loop) -> Strategy Packet -> Execute (Inner Loop) -> Verify (Outer Loop) -> Seal

### D. Pattern 4: Agent Swarm (Parallel Execution)
- **Purpose**: Bulk operations or large disjoint feature sets.
- **Skill**: `plugins/agent-loops/skills/agent-swarm/`
- **Flow**: Partition Work -> Dispatch N Worktrees -> Execute in Isolation -> Verify & Merge -> Seal

### C. Unified Closure Sequence (Phases V-VIII)
Regardless of the chosen execution pattern, **all** loops MUST converge back to the unified closure sequence managed by the Orchestrator and Guardian tools:
1. **Phase V: Orchestrator Retrospective** (`agent_orchestrator.py retro`)
2. **Phase VI: The Technical Seal** (`session-closure` skill → `python3 plugins/sanctuary-guardian/scripts/capture_snapshot.py --type seal`)
3. **Phase VII: Soul Persistence** (`session-closure` skill → `python3 plugins/sanctuary-guardian/scripts/persist_soul.py --snapshot .agent/learning/learning_package_snapshot.md`)
4. **Phase VIII: Session Closure** (`agent_orchestrator.py end` with Git Ops)

## 5. Operational Invariants
- **Git as Source of Truth**: Git diffs (`--stat` and `--name-only`) are the final authority for "what happened."
- **Poka-Yoke**: Successor agents are blocked from holistic action until the previous session's continuity is verified.
- **Sustainability**: Packets must be concise and targeted to prevent steward burnout.
- **Tiered Memory**: Hot cache (boot files) serves 90% of context needs; deep storage (LEARNING/, ADRs/) loaded on demand.
- **Self-Correction**: Failures are data. Phase V uses iterative refinement via the Retrospective until validation passes or max iterations reached.

## 6. Skills Integration Layer (v5.0)

Protocol 128 is operationalized through portable skills in the `plugins/` directory:

| Skill | Phase | Purpose | Location |
| :--- | :--- | :--- | :--- |
| **`orchestrator`** | II-V | Routes tasks to execution patterns, generates retrospectives | `plugins/agent-loops/skills/orchestrator/` |
| **`learning-loop`** | II-IV | Pattern 1: Research & Document | `plugins/agent-loops/skills/learning-loop/` |
| **`red-team-review`**| II-IV | Pattern 2: Audit & Adversarial Review | `plugins/agent-loops/skills/red-team-review/` |
| **`dual-loop`** | II-IV | Pattern 3: Tactical Execution | `plugins/agent-loops/skills/dual-loop/` |
| **`agent-swarm`** | II-IV | Pattern 4: Parallel Execution | `plugins/agent-loops/skills/agent-swarm/` |
| **`session-bootloader`**| I | Session start, debrief ingestion, Iron Check | `plugins/sanctuary-guardian/skills/session-bootloader/` |
| **`session-closure`** | VI-VIII | Technical Seal, Soul Persistence, session end | `plugins/sanctuary-guardian/skills/session-closure/` |
| **`memory-management`** | I, VI, VIII| Tiered memory: hot cache ↔ deep storage | `plugins/memory-management/skills/memory-management/` |

## 7. Document Matrix
| Document | Role | Path |
| :--- | :--- | :--- |
| **ADR 071** | Design Intent | `ADRs/071_protocol_128_cognitive_continuity.md` |
| **Protocol 128** | Constitutional Mandate | `plugins/sanctuary-guardian/resources/protocols/128_Hardened_Learning_Loop.md` |
| **Agent Loops Overview** | Execution Flow Diagram | `plugins/sanctuary-guardian/resources/diagrams/protocol_128_learning_loop.mmd` |
| **Primer** | Rules of Reality | `plugins/sanctuary-guardian/resources/cognitive_primer.md` |
| **Orchestrator Skill** | Loop Router | `plugins/agent-loops/skills/orchestrator/SKILL.md` |
| **Guardian Skill** | Session Boot/Closure | `plugins/sanctuary-guardian/skills/guardian-onboarding/SKILL.md` |

---
**Status:** APPROVED (v5.0)  
**Date:** 2026-02-22
**Authority:** Antigravity (Agent) / Lead (Human)  
**Change Log:**
- v5.0 (2026-02-22): Shifted from monolithic loop to Orchestrator pattern. Redefined 4 execution paths (Learning, Red Team, Dual-Loop, Swarm) mapped to `plugins/agent-loops/`. Mandated closure sequences via Guardian.
- v4.0 (2026-02-11): Added Skills Integration Layer, self-correction patterns, tiered memory invariant
- v3.0 (2025-12-22): Original 10-phase architecture
