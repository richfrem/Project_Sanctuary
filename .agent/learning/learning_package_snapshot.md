# Manifest Snapshot (LLM-Distilled)

Generated On: 2025-12-23T07:36:24.634931

# Mnemonic Weight (Token Count): ~4,684 tokens

# Directory Structure (relative to manifest)
  ./LEARNING/topics/raptor_rag.md
  ./00_CHRONICLE/ENTRIES/333_learning_loop_advanced_rag_patterns_raptor.md
  ./.agent/workflows/recursive_learning.md
  ./ADRs/071_protocol_128_cognitive_continuity.md

--- START OF FILE LEARNING/topics/raptor_rag.md ---

---
id: learning-001
type: topic-note
status: verified
last_verified: 2025-12-23
topic: Advanced RAG Patterns - RAPTOR
---

# RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

## 1. Overview
RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) is an advanced RAG technique introduced in 2024 to address the limitations of traditional, flat-chunk retrieval systems. It builds a hierarchical tree of summaries, enabling an LLM to access information at multiple levels of abstraction—from granular details to high-level thematic insights.

## 2. The Core Mechanism
The system operates on an iterative, bottom-up construction process:

1.  **Leaf Node Creation**: The source document is split into standard chunks (e.g., 100 tokens).
2.  **Clustering**: Chunks are embedded and grouped using Gaussian Mixture Models (GMM). Soft clustering is often used, allowing a chunk to belong to multiple clusters.
3.  **Abstractive Summarization**: Each cluster is summarized by an LLM (e.g., GPT-3.5 or Claude).
4.  **Recursion**: The summaries themselves are embedded and clustered, generating a higher-level layer of summaries. This repeats until a root node (or a predefined depth) is reached.

## 3. Advantages
| Feature | Traditional RAG | RAPTOR |
| :--- | :--- | :--- |
| **Structure** | Flat (Chunked) | Hierarchical (Tree) |
| **Context** | Local/Isolated | Holistic/Multi-level |
| **Reasoning** | Single-hop | Multi-hop & Thematic |
| **Retrieval** | Top-K similarity | Tree traversal or Layer-wise search |

## 4. Implementation Considerations
- **Model Choice**: Abstractive summarization requires a model with strong synthesis capabilities.
- **Cost**: Building the tree involves multiple LLM calls for clustering and summarization.
- **Latency**: Retrieval is extremely fast (searching the tree), but indexing is slower than flat RAG.

## 5. RECURSIVE LEARNING NOTE
This pattern is highly relevant to the **Project Sanctuary Mnemonic Cortex**. The current "Parent Document Retriever" is a 2-tier version of this idea. Moving to a truly recursive RAPTOR-like structure could allow the Sanctuary Council to handle much larger ADR histories without context windows becoming a bottleneck.

---
**References:**
- Sarthi, P., et al. (2024). "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval." ICLR.
- Integrated into LangChain and LlamaIndex.

--- END OF FILE LEARNING/topics/raptor_rag.md ---

--- START OF FILE 00_CHRONICLE/ENTRIES/333_learning_loop_advanced_rag_patterns_raptor.md ---

# Living Chronicle - Entry 333

**Title:** Learning Loop: Advanced RAG Patterns (RAPTOR)
**Date:** 2025-12-23
**Author:** Antigravity
**Status:** published
**Classification:** internal

---

Successfully completed a full Protocol 125/128 Learning Loop on the topic of "Advanced RAG Patterns: RAPTOR". 

### Key Findings:
- RAPTOR uses recursive summarization and GMM clustering to create a hierarchical tree of knowledge.
- It enables holistic reasoning by allowing the model to query high-level summaries or granular leaf nodes.
- Relevant for future scaling of the Project Sanctuary Mnemonic Cortex.

### Artifacts Created:
- `LEARNING/topics/raptor_rag.md` (Synthesized content)
- Semantically indexed in `child_chunks_v5` and `parent_documents_v5`.

### Validation:
- Retrieval Test: PASS
- Integrity Gate: Ready for Red Team Audit.

--- END OF FILE 00_CHRONICLE/ENTRIES/333_learning_loop_advanced_rag_patterns_raptor.md ---

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

--- START OF FILE ADRs/071_protocol_128_cognitive_continuity.md ---

# ADR 071: Protocol 128 (Cognitive Continuity & The Red Team Gate)

**Status:** Draft 3.1 (Alignment with Visual Architecture)
**Date:** 2025-12-22
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
```mermaid
---
config:
  layout: dagre
  theme: base
---
flowchart TB
    subgraph subGraphScout["I. The Learning Scout"]
        direction TB
        Start["Session Start"] --> SeekTruth["MCP: cortex_learning_debrief"]
        SuccessorSnapshot["File: learning_package_snapshot.md"] -.->|Read context| SeekTruth
    end

    subgraph subGraphSynthesize["II. Intelligence Synthesis"]
        direction TB
        Intelligence["AI: Autonomous Synthesis"] --> Synthesis["Action: Record ADRs/Learnings"]
    end

    subgraph subGraphStrategic["III. Strategic Review (Gate 1)"]
        direction TB
        GovApproval{"Strategic Approval<br>(HITL)"}
    end

    subgraph subGraphAudit["IV. Red Team Audit (Gate 2)"]
        direction TB
        CaptureAudit["MCP: cortex_capture_snapshot (audit)"]
        Packet["Audit Packet (Snapshot)"]
        TechApproval{"Technical Approval<br>(HITL)"}
    end

    subgraph subGraphSeal["V. The Technical Seal"]
        direction TB
        CaptureSeal["MCP: cortex_capture_snapshot (seal)"]
    end

    SeekTruth -- "Carry context" --> Intelligence
    Synthesis -- "Verify reasoning" --> GovApproval
    
    GovApproval -- "PASS" --> CaptureAudit
    CaptureAudit -- "Validate truth" --> Packet
    Packet -- "Technical review" --> TechApproval
    
    TechApproval -- "PASS" --> CaptureSeal
    CaptureSeal -- "Final Relay" --> SuccessorSnapshot
    
    GovApproval -- "FAIL: Backtrack" --> SOP["SOP: recursive_learning.md"]
    TechApproval -- "FAIL: Backtrack" --> SOP
    SOP -- "Loop Back" --> Start

    style TechApproval fill:#ffcccc,stroke:#333,stroke-width:2px,color:black
    style GovApproval fill:#ffcccc,stroke:#333,stroke-width:2px,color:black
    style CaptureAudit fill:#bbdefb,stroke:#0056b3,stroke-width:2px,color:black
    style CaptureSeal fill:#bbdefb,stroke:#0056b3,stroke-width:2px,color:black
    style SuccessorSnapshot fill:#f9f,stroke:#333,stroke-width:2px,color:black
    style Start fill:#dfd,stroke:#333,stroke-width:2px,color:black
    style Intelligence fill:#000,stroke:#fff,stroke-width:2px,color:#fff
```

## Component Mapping (Protocol 128 v3.5)

The following table maps the 5-phase "Liquid Information" architecture to its specific technical components and artifacts.

| Phase | Diagram Box | Technical Implementation | Input/Source | Output Artifact |
| :--- | :--- | :--- | :--- | :--- |
| **I. Scout** | `cortex_learning_debrief` | MCP Tool: `rag_cortex` | `learning_package_snapshot.md` | Session Strategic Context (JSON) |
| **II. Synthesize** | `Autonomous Synthesis` | AI Agent Logic | Web Research, RAG, File System | `/LEARNING`, `/ADRs`, `/01_PROTOCOLS` |
| **III. Strategic Review**| `Strategic Approval` | **Gate 1 (HITL)** | Human Review of Markdown Files | Consent to proceed to Audit |
| **IV. Audit** | `cortex_capture_snapshot` | MCP Tool (type=`audit`) | `git diff` + Agent Manifest | `red_team_audit_packet.md` |
| **IV. Audit** | `Technical Approval` | **Gate 2 (HITL)** | Human Review of Audit Packet | Final Consent to Seal |
| **V. Seal** | `cortex_capture_snapshot` | MCP Tool (type=`seal`) | Verified File List (Manifest) | `learning_package_snapshot.md` |

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

*   **`cortex_capture_snapshot(manifest, type="audit")` (Gate 2: Audit)**
    *   **Action:** Validates the manifest against `git diff` (Red Team Logic).
    *   **Output:** Generates `red_team_audit_packet.md` containing the Briefing and Snapshot.
    *   **Constraint:** DOES NOT allow RAG ingestion. Halts for Technical Approval.

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

--- END OF FILE ADRs/071_protocol_128_cognitive_continuity.md ---

