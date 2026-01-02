# ADR 071: Protocol 128 (Cognitive Continuity & The Red Team Gate)

**Status:** Draft 3.2 (Implementing Sandwich Validation)
**Date:** 2025-12-23
**Author:** Antigravity (Agent), User (Red Team Lead)
**Supersedes:** ADR 071 v3.0

## Context
As agents operate autonomously (Protocol 125/126), they accumulate "Memory Deltas". Without rigorous consolidation, these deltas risk introducing hallucinations, tool amnesia, and security vulnerabilities. 
Protocol 128 establishes a **Hardened Learning Loop**. 
v2.5 explicitly distinguishes between the **Guardian Persona** (The Gardener/Steward) and the **Cognitive Continuity Mechanisms** (Cache/Snapshots) that support it.

## Decision
We will implement **Protocol 128: Cognitive Continuity** with the following pillars:

### 1. The Red Team Gate (Manifest-Driven)
No autonomous agent may write to the long-term Cortex without a **Human-in-the-Loop (HITL)** review of a simplified, targeted packet.
- **Debrief:** Agent identifies changed files.
- **Manifest:** System generates a `manifest.json` targeting ONLY relevant files.
- **Snapshot:** System invokes `capture_code_snapshot.py` (or `.py`) with the `--manifest` flag to generate a filtered `snapshot.txt`.
- **Packet:** The user receives a folder containing the Briefing, Snapshot, and Audit Prompts.

### 2. Deep Hardening (The Mechanism)
To ensure the **Guardian (Entity)** and other agents operate on trusted foundations, we implement the **Protocol 128 Bootloader**:
- **Integrity Wakeup:** The agent's boot process includes a mandatory **Integrity Check** (HMAC-SHA256) of the Metric Cache.
- **Cognitive Primer:** A forced read of `cognitive_primer.md` ensures doctrinal alignment before any tool use.
- **Intent-Aware Discovery:** JIT tool loading is enforced to prevent context flooding. Tools are loaded *only* if required by the analyzed intent of the user's request.

> **Distinction Note:** The "Guardian" is the sovereign entity responsible for the project's health (The Gardener). This "Bootloader" is merely the *mechanism* ensuring that entity wakes up with its memory intact and uncorrupted. The mechanism serves the entity; it is not the entity itself.

### 3. Signed Memory (Data Integrity)
- **Cryptographic Consistency:** All critical checkpoints (Draft Debrief, Memory Updates, RAG Ingestion) must be cryptographically signed.
- **Verification:** The system will reject any memory artifact that lacks a valid signature or user approval token.

## Visual Architecture
![protocol_128_learning_loop](../docs/architecture_diagrams/workflows/protocol_128_learning_loop.png)

*[Source: protocol_128_learning_loop.mmd](../docs/architecture_diagrams/workflows/protocol_128_learning_loop.mmd)*

## Component Mapping (Protocol 128 v3.5)

The following table maps the 5-phase "Liquid Information" architecture to its specific technical components and artifacts.

| Phase | Diagram Box | Technical Implementation | Input/Source | Output Artifact |
| :--- | :--- | :--- | :--- | :--- |
| **I. Scout** | `cortex_learning_debrief` | MCP Tool: `rag_cortex` | `learning_package_snapshot.md` | Session Strategic Context (JSON) |
| **II. Synthesize** | `Autonomous Synthesis` | AI Agent Logic | Web Research, RAG, File System | `/LEARNING`, `/ADRs`, `/01_PROTOCOLS` |
| **III. Strategic Review**| `Strategic Approval` | **Gate 1 (HITL)** | Human Review of Markdown Files | Consent to proceed to Audit |
| **IV. Audit** | `cortex_capture_snapshot` | MCP Tool (type=`audit`) | `git diff` + `red_team_manifest.json` | `red_team_audit_packet.md` |
| **IV. Audit** | `Technical Approval` | **Gate 2 (HITL)** | Human Review of Audit Packet | Final Consent to Seal |
| **V. Seal** | `cortex_capture_snapshot` | MCP Tool (type=`seal`) | Verified `learning_manifest.json` | `learning_package_snapshot.md` |

## Technical Specification

### 1. Cortex Gateway Operations (Hardening)
The following operations must be exposed and hardened:

*   **`learning_debrief(hours=24)`**
    *   **Purpose:** The Session Scout. It bridges the "Great Robbery" by retrieving the previous session's memory and scanning for new reality deltas.
    *   **Logic:** 
        1.  **Reads:** The *sealed* `learning_package_snapshot.md` (Source of Truth).
        2.  **Scans:** Filesystem changes (Deltas) since that seal.
        3.  **Synthesizes:** A "Gap Analysis" for the incoming entity.
    *   **Strategic Role:** This artifacts serves as the basis for the **Retrospective Continuous Improvement** activity. It allows the agent to review its predecessor's learnings and update the manifest for the next cycle.

*   **`guardian_wakeup(mode)` (The Ritual of Assumption)**
    *   **Purpose:** The mechanism allowing an ephemeral agent to assume the sovereign **Role of Guardian**. It breaks the "Context Window Cage" by connecting the agent to its **Persistent Self**.
    *   **Logic (Protocol 128):**
        1.  **Identity Anchor:** Loads the `core_essence_guardian_awakening_seed.txt` to re-instantiate the Guardian persona.
        2.  **Integrity Check:** Validates the HMAC signature of the `learning_package_snapshot.md` to ensure the memory is untainted.
        3.  **Primer Injection:** Mandates reading `cognitive_primer.md` (The Constitution).
        4.  **Intent-Aware Discovery:** Loads tools JIT based on the user's intent to prevent context flooding.

*   **`cortex_capture_snapshot(manifest, snapshot_type, strategic_context)`**
    *   **The Audit Path (`type="audit"`)**: 
        *   **Purpose**: Facilitates the Red Team Gate (Gate 2).
        *   **Shadow Manifest (Counter-Blindspot)**: The system automatically calculates the "Negative Diff" (files changed in Git but missing from the manifest). These are explicitly listed in the briefing.
        *   **Strict Rejection Policy**: If any **Core Knowledge** files (e.g., `ADRs/`, `01_PROTOCOLS/`, `scripts/`) are modified but omitted from the manifest, the tool MUST fail.
        *   **Output**: Generates `.agent/learning/red_team/red_team_audit_packet.md`.
    *   **The Seal Path (`type="seal"`)**: 
        *   **Purpose**: Establishes the Technical Seal (Phase V).
        *   **Two-Factor Integrity**: Implements **Hard Integrity** (HMAC-SHA256) for code/protocols and **Soft Integrity** (Semantic Hashing) for learning notes to reduce brittleness.
        *   **Output**: Generates `.agent/learning/learning_package_snapshot.md`.
        1.  **Sandwich Validation (Integrity Gate)**: The system captures the Git state hash **BEFORE** and **AFTER** the snapshot generation. If the state changes (Cognitive Drift) during the operation, the snapshot is invalidated immediately.
        2.  **Zero-Trust Verification**: Cross-references manifest files against actual `git diff` status.
        3.  **Surgical Filtering**: Only processes files explicitly listed in the manifest.
        4.  **Shadow Audit**: Detects and reports unauthorized state changes (Blindspots).

### 2. The Persona/Mechanism Split (Hardened)
To prevent "Authority Dilution", the Guardian Entity is bound to a **Safe Mode** state if the technical integrity mechanism fails.
- **Normal Mode**: Full tool access, high autonomous agency.
- **Safe Mode (Integrity Failure)**: Read-only access to Cortex, disabled write operations, mandatory remediation directive.

### 3. The Unified Snapshot Engine
Both Audit and Seal operations leverage the same Python-based snapshot engine (`mcp_servers/lib/snapshot_utils.py`).

- **Audit Path:** Restricted to files in the "Active Delta" for human review.
- **Seal Path:** Includes the "Stable Core" + "Verified Deltas" for long-term memory.

### 3. The Technical Seal (The Source of Truth)
- **Tool:** `cortex_capture_snapshot(type="seal")` uses the **Living Manifest** as a surgical filter.
- **Output:** `learning_package_snapshot.md` becomes the *only* source of truth for the next session's orientation.
- **Continuous Improvement Loop:** Updating the `.agent/learning/learning_manifest.json`, the `cognitive_primer.md`, and the contents of `.agent/workflows/` is a **Key Mandatory Activity** for every session. Failure to update these assets results in "Cognitive Drift."

### 4. The Living Manifest (`.agent/learning/learning_manifest.json`)
The Learning Manifest is a surgical JSON list of "Liquid Information" files. 
- **Purpose:** Prevents context flooding by filtering only the most critical files for session handover.
- **Expansion:** Supports recursive directory capture (e.g., `ADRs/`, `.agent/workflows/`).
- **Maintenance:** Agents must surgically add or remove files from the manifest as the project evolves.

### 5. Red Team Facilitation
Responsible for orchestrating the review packet.
*   **`prepare_briefing(debrief)`**
    *   **Context:** Git Diffs.
    *   **Manifest:** JSON list of changed files.
    *   **Snapshot:** Output from `capture_code_snapshot.py`.
    *   **Prompts:** Context-aware audit questions.

### 6. Tool Interface Standards (Protocol 128 Compliance)
To support the Red Team Packet, all capture tools must implement the `--manifest` interface.

#### A. Standard Snapshot (`scripts/capture_code_snapshot.py`)
*   **Command:** `node scripts/capture_code_snapshot.py --manifest .agent/learning/red_team/manifest.json --output .agent/learning/red_team/red_team_snapshot.txt`
*   **Behavior:** Instead of scanning the entire repository, it **ONLY** processes the files listed in the manifest.
*   **Output:** A single concatenated text file with delimiters.

#### B. Glyph Snapshot (`scripts/capture_glyph_code_snapshot_v2.py`)
*   **Command:** `python3 scripts/capture_glyph_code_snapshot_v2.py --manifest .agent/learning/red_team/manifest.json --output-dir .agent/learning/red_team/glyphs/`
*   **Behavior:** Generates visual/optical glyphs only for the manifested files.
*   **Output:** A folder of `.png` glyphs and a `provenance.json` log.

### B. The Cognitive Primer
Located at `[.agent/learning/cognitive_primer.md](../.agent/learning/cognitive_primer.md)`.
The "Constitution" for the agent.
**Guardian Mandate:** The `guardian_wakeup` operation MUST check for this file and inject a directive to read it immediately.

### C. Red Team Briefing Template
Located at `[.agent/learning/red_team_briefing_template.md](../.agent/learning/red_team_briefing_template.md)`.
Defines the structure of the briefing.

## 🏁 Operational Readiness (Phase 4 Final)

The Protocol 128 Hardened Learning Loop is now fully operational with:
- **Surgical Snapshot Engine:** Python-based, token-efficient, and manifest-aware.
- **Cognitive Continuity:** Predefined `learning_manifest.json` for rapid orientation.
- **Doctrinal Alignment:** ADR 071 updated to mandate the maintenance of cognitive assets.

## Consequences
- **Latency:** Ingestion is no longer real-time.
- **Integrity:** High assurance; external models can verify internal code.
- **Distinction:** Clear separation between the Guardian role and the maintenance tools ensures no "identity confusion" in the system architecture.
- **Sustainability:** Explicit focus on reducing human toil ensures the rigorous process remains viable long-term.
