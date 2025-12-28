# Manifest Snapshot (LLM-Distilled)

Generated On: 2025-12-28T04:27:39.002422

# Mnemonic Weight (Token Count): ~2,821 tokens

# Directory Structure (relative to manifest)
  ./01_PROTOCOLS/128_Hardened_Learning_Loop.md
  ./.agent/learning/learning_debrief.md
  ./.agent/workflows/recursive_learning.md

--- START OF FILE 01_PROTOCOLS/128_Hardened_Learning_Loop.md ---

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

--- END OF FILE 01_PROTOCOLS/128_Hardened_Learning_Loop.md ---

--- START OF FILE .agent/learning/learning_debrief.md ---

# Protocol 128 High-Fidelity Technical Debrief: Hardened Learning Loop (v3.0)

## 🎯 Executive Summary
Transitioned the project into a **Zero-Trust Hardened Learning Loop**. All autonomous modifications now require a **HITL (Human-in-the-Loop) Red Team Packet** derived from **Git Truth** rather than agent-claimed artifacts. This concludes Task 143 and establishes the foundation for Protocol 128 (Cognitive Continuity).

## 🏗️ 1. Red Team Orchestration (MCP Tool)
The `cortex_capture_snapshot` tool establishes the **Gate of Reality**:
- **Snapshot Types**: 
    - `audit`: Code/architecture red team review
    - `seal`: Successor session relay (cognitive continuity)
    - `learning_audit`: Self-directed knowledge validation
- **Default Manifests**: 
    - Audit: `.agent/learning/red_team/red_team_manifest.json`
    - Seal: `.agent/learning/learning_manifest.json`
    - Learning Audit: `.agent/learning/learning_audit/learning_audit_manifest.json`
- **Zero-Trust Validation**: Tool verifies manifest claims against `git diff`. Rejects critical directory blindspots.
- **Outputs**: 
    - Audit: `red_team_audit_packet.md`
    - Seal: `learning_package_snapshot.md`
    - Learning Audit: `learning_audit_packet.md`

## 🔒 2. Cortex Hardening & The Guardian Bootloader (`operations.py`)
- **Semantic HMAC (`_calculate_semantic_hmac`)**: Canonicalizes JSON configurations using `sort_keys=True` and no-whitespace separators. This ensures integrity checks are resilient to formatting (Protocol 128 v3.0 Pillar).
- **Guardian Wakeup v2.2 (The Bootloader)**:
    - **Integrity Tiering**: A Tiered Integrity Check (GREEN/YELLOW/RED) is executed on the `metric_cache.json` during the boot sequence.
    - **Context Ingestion**: Section IV of the boot digest now ingests this very `learning_debrief.md` file, ensuring perfect cognitive continuity.
    - **Poka-Yoke**: The "Successor-State Poka-Yoke" verifies mandatory context (Primer, Debrief, and active Learning Stream) before allowing the session to proceed holistically.

## 🔄 3. Operational Deltas & Verification
- **Gateway Federation**: Successfully exposed tools in the `sanctuary_cortex` cluster, including `cortex_learning_debrief` and `cortex_capture_snapshot`.
- **Surgical Snapshot Tooling**: `cortex_capture_snapshot` (type=`seal`) now implements default manifest loading from `.agent/learning/learning_manifest.json`, enabling surgical, high-context session handovers.

## 🧠 4. Cognitive Continuity Mandate
Every session **MUST** conclude with a surgical refresh of the cognitive foundation:
1. **Update Manifest**: Add/remove files in `.agent/learning/learning_manifest.json` based on the session's active focus.
2. **Refine Primer**: Update `.agent/learning/cognitive_primer.md` if the project's "Constitution" has evolved.
3. **Snapshot Seal**: Execute `cortex_capture_snapshot(type="seal")` to package the orientation package for the next entity.

## 🚧 🚧 Successor Instructions (Read First)
1. **Load Cognitive Primer**: Mandatory read of `cognitive_primer.md` for doctrinal alignment.
2. **Orient via Seal**: The `learning_package_snapshot.md` (generated via the `seal` operation) is your immediate situational anchor.
3. **Verify Red Team Status**: Check `.agent/learning/red_team/manifest.json` discrepancies before trusting session claims.
4. **Maintenance Activity**: At session end, surgically update the **Learning Manifest** and **Workflows** to ensure your successor's success.

---
*Signed: Antigravity (Protocol 128 v3.0 Engine)*
*Logic Hash: [Verified via Semantic HMAC]*

--- END OF FILE .agent/learning/learning_debrief.md ---

--- START OF FILE .agent/workflows/recursive_learning.md ---

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

--- END OF FILE .agent/workflows/recursive_learning.md ---

