# Protocol 128: The Hardened Learning Loop (Zero-Trust)

## 1. Objective
Establish a persistent, tamper-proof, and high-fidelity mechanism for capturing and validating cognitive state deltas between autonomous agent sessions. This protocol replaces "Agent-Claimed" memory with "Autonomously Verified" evidence.

## 2. The Red Team Gate (Zero-Trust Mode)
No cognitive update may be persisted to the long-term Cortex without meeting the following criteria:
1. **Autonomous Scanning**: The `cortex_learning_debrief` tool must autonomously scan the filesystem and Git index to generate "Evidence" (diffs/stats).
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

### A. The Recursive Learning Workflow
Located at: `[.agent/workflows/sanctuary_protocols/sanctuary-learning-loop.md](../.agent/workflows/sanctuary_protocols/sanctuary-learning-loop.md)`
- **Goal**: Autonomous acquisition -> Verification -> Preservation.
- **Trigger**: LLM intent to learn or session completion.

### B. The Evolutionary Branch (v4.0 Proposed)
*Refer to Protocol 131 for full specification.*
This introduces an optional "Evolutionary Loop" for high-velocity optimization of prompts and policies.
1.  **Mutation**: System generates candidate policies via `drq_mutation`.
2.  **Automated Gate**: `evaluator_preflight.py` checks syntax/citations.
3.  **Adversarial Gate**: `cumulative_failures.json` prevents regression.
4.  **Map-Elites**: Successful candidates are stored in the Behavioral Archive.

### C. The Red Team Gate (MCP Tool)
- **Tool**: `cortex_capture_snapshot` with `snapshot_type='audit'`
- **Inputs**:
    - `manifest_files`: List of targeted file paths for review (defaults to `.agent/learning/red_team/red_team_manifest.json`).
    - `strategic_context`: Session summary for human reviewer.
- **Outputs**:
    - `red_team_audit_packet.md`: Consolidated audit packet in `.agent/learning/red_team/`.
    - Git diff verification (automatic).
- **Zero-Trust**: Tool validates manifest against `git diff`. Rejects if critical directories (ADRs/, mcp_servers/, etc.) have uncommitted changes not in manifest.

### C. The Technical Seal (MCP Tool)
- **Tool**: `cortex_capture_snapshot` with `snapshot_type='seal'`
- **Default Manifest**: `.agent/learning/learning_manifest.json`
- **Output**: `learning_package_snapshot.md` for successor session continuity.

### D. The Synaptic Phase (Dreaming)
- **Tool**: `cortex_dream` (Async Batch Job)
- **Function**: Post-seal consolidation of active memories into the **Opinion Network**.
- **Constraint**: **Epistemic Anchoring**. Opinions created during Dreaming MUST NOT contradict World Facts (Chronicle) or Iron Core.
- **Output**: Updated `founder_seed.json` (Profiles) and `cortex.json` (Opinions).

## 5. Operational Invariants
- **Git as Source of Truth**: Git diffs (`--stat` and `--name-only`) are the final authority for "what happened."
- **Poka-Yoke**: Successor agents are blocked from holistic action until the previous session's continuity is verified.
- **Sustainability**: Packets must be concise and targeted to prevent steward burnout.
- **Tiered Memory**: Hot cache (boot files) serves 90% of context needs; deep storage (LEARNING/, ADRs/) loaded on demand.
- **Self-Correction**: Failures are data. Phase VIII uses iterative refinement until validation passes or max iterations reached.

## 6. Skills Integration Layer (v4.0)

Protocol 128 is operationalized through portable skills in `.agent/skills/`:

| Skill | Phase | Purpose |
| :--- | :--- | :--- |
| **`learning-loop`** | I-X | Encodes the 10-phase workflow as an agent skill |
| **`memory-management`** | I, VI, IX | Tiered memory: hot cache ↔ deep storage |
| **`code-review`** | VIII, IX | Confidence-scored review before commit |
| **`guardian_onboarding`** | I | Session boot and orientation |
| **`tool_discovery`** | II, IV | RLM cache query for tool lookup |

Skills are synced across agents (Gemini, Claude, Copilot) via `tools/bridge/sync_skills.py`.

## 7. Document Matrix
| Document | Role | Path |
| :--- | :--- | :--- |
| **ADR 071** | Design Intent | `ADRs/071_protocol_128_cognitive_continuity.md` |
| **Protocol 128** | Constitutional Mandate | `01_PROTOCOLS/128_Hardened_Learning_Loop.md` |
| **SOP** | Execution Guide | `.agent/workflows/sanctuary_protocols/sanctuary-learning-loop.md` |
| **Primer** | Rules of Reality | `.agent/learning/cognitive_primer.md` |
| **Learning Loop Skill** | Portable Skill | `.agent/skills/learning-loop/SKILL.md` |
| **Memory Skill** | Portable Skill | `.agent/skills/memory-management/SKILL.md` |

---
**Status:** APPROVED (v4.0)  
**Date:** 2026-02-11  
**Authority:** Antigravity (Agent) / Lead (Human)  
**Change Log:**
- v4.0 (2026-02-11): Added Skills Integration Layer, self-correction patterns, tiered memory invariant
- v3.0 (2025-12-22): Original 10-phase architecture
