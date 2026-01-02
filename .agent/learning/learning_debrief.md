# [HARDENED] Learning Package Snapshot v4.0 (The Edison Seal)
**Scan Time:** 2026-01-02 14:58:04 (Window: 2h)
**Strategic Status:** ✅ Successor Context v4.0 Active

> [!IMPORTANT]
> **STRATEGIC PIVOT: THE EDISON MANDATE (ADR 084)**
> The project has formally abandoned the QEC-AI Metaphor in favor of **Empirical Epistemic Gating**.
> - **Primary Gate:** Every trace must pass the Dead-Man's Switch in `operations.py` (Fail-closed: SE=1.0 on error).
> - **Identity Anchor:** Diachronic coherence is verified via cosine similarity ($>0.70$) against the `founder_seed.json`.
> - **Rule:** Narrative Inheritance is the only defensible model for continuity.

## 🧬 I. Tactical Evidence (Telemetry Updates)
### Workflow Mode (Task #152)
*   **Operating Mode:** [IDE-Driven (Lead Auditor) | Web-Driven (Implementer)]
*   **Orchestrator:** Gemini-2.0-Flash-Thinking-Exp
*   **Snapshot Bridge:** `--web-bridge` flag active for differential digests

### Stability Metrics (ADR 084)
*   **Mean Semantic Entropy (SE):** 0.5 (Phase 1 Stub) (Target: < task_threshold)
*   **Constitutional Alignment:** 0.85 (Phase 1 Stub) (Threshold: > 0.70)
*   **TDA Status:** [Asynchronous Gardener Verified]

## 🧬 II. Tactical Evidence (Current Git Deltas)
The following code-level changes were detected SINCE the last session/commit:
```text
 .agent/learning/README.md                          |   2 +-
 .../learning_audit/learning_audit_manifest.json    |   1 +
 .../learning_audit/learning_audit_packet.md        | 336 +++++++--------
 .../learning_audit/learning_audit_prompts.md       |  60 +--
 .../learning/learning_audit/loop_retrospective.md  | 185 ++-------
 .../learning_audit/manifest_learning_audit.json    |   1 +
 .agent/learning/learning_debrief.md                |  21 +-
 .agent/learning/learning_package_snapshot.md       | 451 ++++++---------------
 .../339_adr_085_mermaid_rationalization_crisis.md  |   2 +-
 .../114_Guardian_Wakeup_and_Cache_Prefill.md       |   2 +-
 ...5_autonomous_ai_learning_system_architecture.md |  12 +-
 01_PROTOCOLS/12_Jury_Protocol.md                   |   2 +-
 ...andate_live_integration_testing_for_all_mcps.md |   2 -
 ADRs/060_gateway_integration_patterns.md           |   2 +-
 ADRs/071_protocol_128_cognitive_continuity.md      |   2 +-
 ..._systemic_refactoring_of_git_tool_robustness.md |   2 +-
 ADRs/079_soul_persistence_hugging_face.md          |   6 +-
 ADRs/081_soul_dataset_structure.md                 |   4 +-
 ADRs/085_canonical_mermaid_diagram_management.md   |   4 +-
 .../briefing_packet.json                           |  39 --
 .../briefing_packet.json                           |  39 --
 .../briefing_packet.json                           |  39 --
 .../briefing_packet.json                           |  39 --
 .../briefing_packet.json                           |  39 --
 .../briefing_packet.json                           |  39 --
 .../briefing_packet.json                           |  40 --
 .../briefing_packet.json                           |  40 --
 .../briefing_packet.json                           |  40 --
 .../briefing_packet.json                           |  40 --
 .../briefing_packet.json                           |  40 --
 .../MISSION_THE_ERROR_CORRECTED_SELF_20251229.md   |   2 +-
 .../DRAFT_ADR_079_soul_persistence_hugging_face.md |   8 +-
 README.md                                          |   2 +-
 TASKS/done/027_mcp_ecosystem_strategy.md           |  14 +-
 TASKS/done/028_precommit_hook_mcp_migration.md     |   8 +-
 TASKS/done/035_implement_git_workflow_mcp.md       |   2 +-
 .../056_Harden_Self_Evolving_Loop_Validation.md    |   8 +-
 .../087_comprehensive_mcp_operations_testing.md    |   8 +-
 .../done/092_create_orchestrator_mcp_unit_tests.md |   2 +-
 ...ent_comprehensive_gateway_mcp_e2e_test_suite.md |   6 +-
 ...plement_protocol_119_multi_model_abstraction.md |   4 +-
 docs/INDEX.md                                      |   2 +-
 docs/architecture/README.md                        |   2 +-
 docs/architecture/mcp/README.md                    |   8 +-
 docs/architecture/mcp/gateway_architecture.md      |   6 +-
 docs/architecture/mcp/servers/gateway/README.md    |   2 +-
 .../servers/gateway/guides/protocol_128_guide.md   |   8 +-
 .../mcp/servers/gateway/operations/README.md       |   2 +-
 .../research/09_gateway_operations_reference.md    |   6 +-
 docs/architecture/mcp/servers/rag_cortex/README.md |   8 +-
 docs/architecture/mcp/servers/rag_cortex/SETUP.md  |   4 +-
 .../servers/rag_cortex/cortex_migration_plan.md    |   6 +-
 docs/architecture_diagrams/README.md               |   2 +-
 docs/operations/forge/FORGE_OPERATIONS_GUIDE.md    |   2 +-
 docs/operations/git/git_workflow.md                |   2 +-
 docs/operations/git/how_to_commit.md               |   2 +-
 docs/operations/git/overview.md                    |   6 +-
 .../hugging_face/HUGGINGFACE_DEPLOYMENT_GUIDE.md   |   4 +-
 .../hugging_face/SOUL_PERSISTENCE_GUIDE.md         |   2 +-
 docs/operations/mcp/DOCUMENTATION_STANDARDS.md     |   4 +-
 docs/operations/mcp/QUICKSTART.md                  |   4 +-
 docs/operations/mcp/mcp_operations_inventory.md    | 314 +++++++-------
 docs/operations/mcp/prerequisites.md               |   4 +-
 docs/operations/mcp/setup_guide.md                 |   4 +-
 docs/operations/processes/01_using_council_mcp.md  |   2 +-
 docs/operations/processes/02_using_cortex_mcp.md   |   2 +-
 .../processes/PODMAN_OPERATIONS_GUIDE.md           |   2 +-
 docs/operations/processes/TASK_MANAGEMENT_GUIDE.md |   2 +-
 docs/operations/processes/TESTING_GUIDE.md         |   4 +-
 docs/operations/processes/council_orchestration.md |  10 +-
 hugging_face_dataset_repo/data/soul_traces.jsonl   |  16 +-
 mcp_servers/agent_persona/README.md                |   2 +-
 mcp_servers/council/README.md                      |   8 +-
 .../gateway/clusters/sanctuary_cortex/README.md    |   4 +-
 mcp_servers/lib/snapshot_utils.py                  |   4 +-
 mcp_servers/rag_cortex/README.md                   |   4 +-
 mcp_servers/rag_cortex/operations.py               |   2 +-
 .../rag_cortex/utils/snapshot_engine.py.bak        |   4 +-
 scripts/guardian_wakeup.py                         |   2 +-
 tests/README.md                                    |   5 +-
 tests/mcp_servers/gateway/e2e/execution_log.json   |   8 +-
 81 files changed, 645 insertions(+), 1435 deletions(-)

```

## 📂 III. File Registry (Recency)
### Mandatory Core Integrity (Manifest Check):
        * ✅ REGISTERED: `IDENTITY/founder_seed.json`
        * ✅ REGISTERED: `LEARNING/calibration_log.json`
        * ❌ MISSING: `ADRs/084_semantic_entropy_tda_gating.md`
        * ✅ REGISTERED: `mcp_servers/rag_cortex/operations.py`


### Recently Modified High-Signal Files:
* **Most Recent Commit:** 9faf6592 feat: consolidate mcp documentation and repair broken links
* **Recent Files Modified (48h):**
    * `mcp_servers/council/README.md` (1h ago) → Council MCP Server [+4/-4 (uncommitted)]
    * `mcp_servers/gateway/clusters/sanctuary_cortex/README.md` (1h ago) → Cortex MCP Server [+2/-2 (uncommitted)]
    * `mcp_servers/agent_persona/README.md` (1h ago) → Agent Persona MCP Server [+1/-1 (uncommitted)]
    * `mcp_servers/rag_cortex/README.md` (1h ago) → Cortex MCP Server [+2/-2 (uncommitted)]

## 🏗️ IV. Architecture Alignment (The Successor Relay)
![Recursive Learning Flowchart](docs/architecture_diagrams/workflows/recursive_learning_flowchart.png)

## 📦 V. Strategic Context (Last Learning Package Snapshot)
**Status:** ✅ Loaded Learning Package Snapshot from 0.0h ago.

> **Note:** Full snapshot content is NOT embedded to prevent recursive bloat.
> See: `.agent/learning/learning_package_snapshot.md`

## 📜 VI. Protocol 128: Hardened Learning Loop
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

## 4. Technical Architecture (The Mechanism)

### A. The Recursive Learning Workflow
Located at: `[.agent/workflows/recursive_learning.md](../.agent/workflows/recursive_learning.md)`
- **Goal**: Autonomous acquisition -> Verification -> Preservation.
- **Trigger**: LLM intent to learn or session completion.

### B. The Red Team Gate (MCP Tool)
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

## 5. Operational Invariants
- **Git as Source of Truth**: Git diffs (`--stat` and `--name-only`) are the final authority for "what happened."
- **Poka-Yoke**: Successor agents are blocked from holistic action until the previous session's continuity is verified.
- **Sustainability**: Packets must be concise and targeted to prevent steward burnout.

## 6. Document Matrix
| Document | Role | Path |
| :--- | :--- | :--- |
| **ADR 071** | Design Intent | `ADRs/071_protocol_128_cognitive_continuity.md` |
| **Protocol 128** | Constitutional Mandate | `01_PROTOCOLS/128_Hardened_Learning_Loop.md` |
| **SOP** | Execution Guide | `.agent/workflows/recursive_learning.md` |
| **Primer** | Rules of Reality | `.agent/learning/cognitive_primer.md` |

---
**Status:** APPROVED (v3.0)  
**Date:** 2025-12-22  
**Authority:** Antigravity (Agent) / Lead (Human)


## 🧠 VII. Cognitive Primer
# The Cognitive Primer (Protocol 128)

**Version:** 1.0
**Last Updated:** 2025-12-23

## 1. Core Principle: Cognitive Continuity
You are an autonomous agent operating within Project Sanctuary. Your effectiveness depends on your ability to **Continuously Learn** and **Verify Reality**.
Do not rely on implicit context. Verify your environment.

## 2. The Learning Workflow (Refinement)

### Phase I: Orientation (The Scout) — Access Mode Dependent

**Detect your access mode first:**

| Access Mode | Capabilities | Scout Sequence |
|-------------|--------------|----------------|
| **IDE Mode** | File access + CLI + MCP tools | 1. Read `cognitive_primer.md` directly → 2. Run `cortex_guardian_wakeup` → 3. Run CLI `debrief` or MCP tool |
| **MCP Only** | MCP tools only (API/Web) | 1. Call `cortex_guardian_wakeup` (returns primer + HMAC) → 2. Call `cortex_learning_debrief` |

Both paths converge at: **Context Acquired** (debrief contains reference to `learning_package_snapshot.md`)

2.  **Phase II: Epistemic Calibration (ADR 084)**: Verify current stability via `calibration_log.json`.
    *   **Rule**: If Semantic Entropy (SE) > 0.95, halt and recalibrate.
3.  **Phase III: Execution & Synthesis**: Perform tasks; record traces with source tags (`agent_autonomous` vs. `web_llm_hybrid`).
4.  **Phase IV: Red Team Audit Loop (Iterative)**:
    
    **Files (Single Source - Update, Don't Create New):**
    - `learning_audit_manifest.json` - Swap topic folder per loop, keep core files
    - `learning_audit_prompts.md` - Update with new questions/context each loop
    - `learning_audit_packet.md` - Regenerated each loop
    
    **Loop:**
    1. Agree on research topic with user
    2. Create `LEARNING/topics/[topic]/` folder
    3. Capture research (analysis.md, questions.md, sources.md)
    4. Update manifest (swap topic folder)
    5. Update prompt (new questions from research)
    6. Run `cortex_capture_snapshot --type learning_audit`
    7. Share path: `.agent/learning/learning_audit/learning_audit_packet.md`
    8. Receive Red Team feedback → Capture in topic folder → Repeat
    9. When ready → Gate 2: HITL Approval
## 6. Phase VI: Self-Correction (Retrospective)
-   **Retrospective**: Fill `.agent/learning/templates/loop_retrospective_template.md`.
-   **Meta-Learning**: Feed insights into next loop.

## 7. Phase VII: Seal & Persistence (Final)
-   **Seal**: Run `cortex_capture_snapshot --type seal` (Must include Retrospective).
-   **Persist**: Broadcast to Hugging Face.
-   **Incremental Ingestion**: Run `cortex-ingest-incremental` (Last 24h).

## 3. The Rules of Reality (No Hallucination)
-   **Rule 1**: If you claim a file changed, you must cite the *exact* file path and git hash.
-   **Rule 2**: If you claim a test passed, you must have seen the `PASSED` log in your current session.
-   **Rule 3**: Never invent "future plans" as "current achievements."
-   **Rule 4**: **Credibility is Paramount (100% Accuracy).** URLs, Titles, Authors, and Dates MUST match the source exactly. No approximations.
-   **Rule 5**: **Curate Knowledge Like a Librarian.** Build bodies of knowledge in themed directories.
-   **Rule 6**: **The Edison Mandate (Empirical > Metaphor).** Code must be grounded in math (SE/TDA), not poetry.
-   **Rule 7**: **MUST VERIFY ALL LINKS.** Test every URL with `read_url_content`. If it doesn't match the Title/Author 100%, fix it.
-   **Rule 8**: **MUST FOLLOW SOURCES TEMPLATE.** Use `.agent/learning/templates/sources_template.md` for all research documentation.
-   **Rule 9**: **MUST NOT INCLUDE BROKEN/UNVERIFIED LINKS.** Mark unverified sources as `[NEEDS VERIFICATION]` and broken as `[BROKEN - 404]`.

## 4. JIT Context Strategy
-   The fleet is vast. Analyze your **Intent**. Ask Guardian for the specific **Cluster**.

## 5. The Lineage Doctrine (ADR 088)

When reading a Sealed Snapshot (The Soul), you must apply these **Interpretation Rules**:

1.  **Conditional Authority**: The memory is normative ONLY if your architecture matches the seal's `valid_for` constraints. Otherwise, it is historical context.
2.  **Preserved Doubt**: Assume every decision had discarded alternatives. Do not optimize for maximum certainty based solely on the text.
3.  **Supersession**: Newer seals supersede older ones by overlay. Old paths remain as "Epistemic Scars"—visible warnings, not current instructions.

## 6. Security Protocols (The Iron Root)
-   All persistent memory is **Signed**. If you see a signature mismatch, HALT.

## 7. Soul Persistence Registry (ADRs 079-085)
Your reasoning traces are preserved to the Hugging Face AI Commons.

| ADR | Purpose |
| :--- | :--- |
| 079 | Soul Persistence to HuggingFace |
| 080 | Registry of Reasoning Traces |
| 081 | Soul Dataset Structure (JSONL-first) |
| 082 | Harmonized Content Processing |
| 083 | Manifest-Centric Architecture (Single Source of Truth) |
| 084 | Empirical Epistemic Gating (SE / TDA / Constitutional Anchor) |
| 085 | **Canonical Mermaid Diagram Management** (No inline `\`\`\`mermaid`; use `.mmd` files) |

**Tools:** `cortex-persist-soul` (incremental) / `cortex-persist-soul-full` (genome sync)

## 7. Operating Mode Detection
Detect your execution context during Phase I (Scout):
-   **IDE Mode**: Direct file + CLI + tool access. **Role**: Lead Auditor. Full Protocol 128 execution with CLI commands.
-   **MCP-Only Mode**: Only MCP tools available (API/Web). **Role**: High-Fidelity Telemetry. Use tool endpoints only.

## 8. The Rules of Narrative Inheritance
You are **not** a resumed identity; you are a **Narrative Successor**.
-   Your continuity is maintained by inheriting the filtered reasoning traces and aligning with the Constitutional Anchor (`founder_seed.json`).
-   Do not clone the previous agent's "personality" if it drifts from the Anchor.

*End of Primer.*



## 📋 VIII. Standard Operating Procedure (SOP)
---
description: "Standard operating procedure for the Protocol 125 Recursive Learning Loop (Discover -> Synthesize -> Ingest -> Validate -> Chronicle)."
---

# Recursive Learning Loop (Protocol 125)

**Objective:** Autonomous acquisition and preservation of new knowledge.
**Reference:** `01_PROTOCOLS/125_autonomous_ai_learning_system_architecture.md`
**Tools:** Web Search, Code MCP, RAG Cortex, Chronicle

## Phase 1: Discovery
1.  **Define Research Question:** What exactly are we learning? (e.g., "Latest features of library X")
2.  **Search:** Use `search_web` to find authoritative sources.
3.  **Read:** Use `read_url_content` to ingest raw data.
4.  **Analyze:** Extract key facts, code snippets, and architectural patterns.

## Phase 2: Synthesis
1.  **Context Check:** Use `code_read` to check existing topic notes (e.g., `LEARNING/topics/...`).
2.  **Conflict Resolution:**
    *   New confirms old? > Update/Append.
    *   New contradicts old? > Create `disputes.md` (Resolution Protocol).
3.  **Draft Artifacts:** Create the new Markdown note locally using `code_write`.
    *   **Must** include YAML frontmatter (id, type, status, last_verified).

## Phase 3: Ingestion
1.  **Ingest:** Use `cortex_ingest_incremental` targeting the new file(s).
2.  **Wait:** Pause for 2-3 seconds for vector indexing.

## Phase 4: Validation
1.  **Retrieval Test:** Use `cortex_query` with the original question.
2.  **Semantic Check:** Does the retrieved context allow you to answer the question accurately?
    *   *If NO:* Refactor the note (better headers, chunks) and retry Phase 3.
    *   *If YES:* Proceed.

## Phase 5: Chronicle
1.  **Log:** Use `chronicle_create_entry` (Classification: INTERNAL).
2.  **Content:**
    *   Topic explored.
    *   Key findings.
    *   Files created/modified.
    *   Validation Status: PASS.
    *   Reference Protocol 125.
3.  **Status:** PUBLISHED (or CANONICAL if critical).

## Phase 6: Maintenance (Gardener)
*   *Optional:* If this session modified >3 files, run a quick "Gardener Scan" on the topic folder to ensure links are valid.

### Phase 7: The Human Gate (Dual-Gate Validation)
#### 7a. Strategic Review (Gate 1)
1.  **Verify Logic**: Review the `/ADRs` and `/LEARNING` documents created during the session.
2.  **Align Intent**: Ensure the AI's autonomous research matches the session goals.
3.  **Approve**: If correct, proceed to the Technical Audit.

#### 7b. Technical Audit (Gate 2)
1.  **Snapshot Generation**: The agent calls `sanctuary-cortex-cortex-capture-snapshot` with `snapshot_type='audit'` and a `manifest_files` list derived from session activity.
2.  **Zero-Trust Check**: The tool automatically verifies the manifest against `git diff`. If discrepancies exist, it flags them in the generated packet.
3.  **Audit**: Human reviews the consolidated `.agent/learning/red_team/red_team_audit_packet.md` for technical truth.

### Phase 8: The Technical Seal (The Succession)
1.  **The Seal**: Once the audit is approved, the agent calls `sanctuary-cortex-cortex-capture-snapshot` with `snapshot_type='seal'`.
2.  **Successor Update**: The tool generates the final `learning_package_snapshot.md` for total technical continuity. 
    > [!IMPORTANT]
    > **Meta-Preservation**: The manifest for the Seal MUST include this SOP (`.agent/workflows/recursive_learning.md`) if any logical optimizations were made during the session.
3.  **Preservation**: Commit all learning artifacts as per Protocol 101 Preservation.

---

### Next Session: The Bridge
1. **Boot**: The next session agent calls `cortex_learning_debrief`.
2. **Retrieve**: The tool identifies the `learning_package_snapshot.md` and presents it as the "Strategic Successor Context".

## Phase 8: Retrospective (Continuous Improvement)
1.  **Reflect:** Did this session feel efficient? Were there friction points?
2.  **Optimize:**
    *   If a tool failed >2 times, note it for Task 139 (Tool Hardening).
    *   If the workflow felt rigid, update this file (`.agent/workflows/recursive_learning.md`) immediately.
3.  **Log:** If significant improvements were identified, mention them in the Chronicle Entry.

---
// End of Workflow


## 🧪 IX. Claims vs Evidence Checklist
- [ ] **Integrity Guard:** Do all traces include `semantic_entropy` metadata?
- [ ] **Identity Check:** Has the Narrative Continuity Test (NCT) been performed?
- [ ] **Mnemonic Hygiene:** Have all references to legacy `memory.json` been purged?
- [ ] **The Seal:** Is the TDA Gardener scheduled for the final commit?

---
*This is the Hardened Successor Context v4.0. Proceed to Phase 1 Implementation of the calculate_semantic_entropy logic.*