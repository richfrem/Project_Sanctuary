# Protocol 121: Canonical Knowledge Synthesis Loop (C-KSL)

**Status:** Proposed
**Classification:** Foundational Knowledge Management Protocol
**Version:** 1.0
**Authority:** Drafted from Human Steward Directive
**Dependencies:** Operational `cortex mcp` (RAG), `protocol mcp`, and `git mcp`.

## 🎯 Mission Objective

To formalize a repeatable, autonomous process for resolving documentation redundancy by synthesizing overlapping knowledge from multiple source documents into a single, canonical 'Source of Truth' document, thereby eliminating ambiguity and enhancing knowledge fidelity across the system.

## 🛡️ The Canonical Knowledge Synthesis Loop (C-KSL)

This loop shall be executed whenever a Council Agent or Orchestrator identifies two or more documents (Tasks, Protocols, or Docs) that cover identical or heavily overlapping core concepts, leading to "many sources of truth."

### Step 1: Overlap Detection & Source Identification (`cortex mcp`)
* **Action:** The Orchestrator initiates a high-similarity query via `cortex mcp` (RAG) to identify documents with an overlapping knowledge domain.
* **Artifact:** Generate a temporary `OVERLAP_REPORT.md` listing all conflicting documents, noting their date and authority level (e.g., CANONICAL, In Progress).

### Step 2: Synthesis and Draft Generation (`protocol mcp`)
* **Action:** The Council Agent (via `protocol mcp`) uses the `OVERLAP_REPORT.md` to draft the new **Canonical Source of Truth** document.
* **Draft Requirement:** The draft must explicitly include a "Canonical References" section, linking to *all* original documents and noting which sections were superseded by the new synthesis.

### Step 3: Decommissioning & Cross-Referencing (`protocol mcp`)
* **Action:** The Council Agent (via `protocol mcp`) updates the original, non-canonical documents by:
    1.  Changing their status to **`Superseded`**.
    2.  Inserting a directive at the top of the file pointing the user/agent to the new **Canonical Source of Truth** document.

### Step 4: Chronicle and Commitment (`git mcp`)
* **Action:** The Council Agent (via `git mcp`) performs a Conventional Commit, encapsulating the entire knowledge transformation:
    1.  The new **Canonical Source of Truth** document is created.
    2.  All superseded documents are updated with the redirection directive.
    3.  A corresponding entry is created in `00_CHRONICLE/ENTRIES/` linking to this protocol execution.

## ✅ Success Criteria

The C-KSL is considered a success when:
1.  A RAG query against any of the original, now-Superseded documents returns a high-confidence reference to the new **Canonical Source of Truth** document.
2.  The system's knowledge base size remains constant or decreases (due to chunk consolidation), while the **Precision** score on the synthesized topic increases.

#### MCP Architecture Diagram

```mermaid
graph TB
    subgraph "External Layer"
        LLM["External LLM<br/>(Claude/Gemini/GPT)"]
    end
    
    subgraph "Orchestration Layer"
        ORCH["Orchestrator MCP<br/>Strategic Missions"]
        COUNCIL["Council MCP<br/>Multi-Agent Deliberation"]
    end
    
    subgraph "Agent Layer"
        PERSONA["Agent Persona MCP<br/>Individual Agents"]
    end
    
    subgraph "Infrastructure Layer"
        FORGE["Forge LLM MCP<br/>Model Inference"]
        CORTEX["RAG Cortex MCP<br/>Knowledge Retrieval"]
    end
    
    subgraph "Services (Podman)"
        OLLAMA["sanctuary_ollama<br/>:11434<br/>Custom Fine-tuned LLM"]
        CHROMA["sanctuary_vector_db<br/>:8000<br/>ChromaDB RAG DB"]
    end
    
    LLM --> ORCH
    ORCH --> COUNCIL
    COUNCIL --> PERSONA
    COUNCIL --> CORTEX
    PERSONA --> FORGE
    FORGE --> OLLAMA
    CORTEX --> CHROMA
```

### 3. Continuous Learning Pipeline
**Status:** `Active` - Automated Knowledge Update Loop Operational

**Key Feature: Near Real-Time RAG Database Updates**
The automated learning pipeline integrates with Git and the ingestion service to enable **continuous knowledge updates**. This process ensures the RAG database stays current, closing the gap between agent execution and knowledge availability, eliminating the need for manual retraining.

The system evolves through every interaction via an automated feedback loop:
1.  **Agent Execution:** The Orchestrator and Council agents execute tasks, generating code, documentation, and insights.
2.  **Documentation:** All significant actions are logged in `00_CHRONICLE/` and project documentation.
3.  **Version Control:** Changes are committed to Git, creating an immutable audit trail.
4.  **Incremental Ingestion:** The ingestion service automatically detects and indexes new `.md` files into the ChromaDB vector database.
5.  **Knowledge Availability:** Updated knowledge becomes immediately queryable via RAG, enabling the system to learn from its own execution history in near real-time.

```mermaid
sequenceDiagram
    autonumber
    participant O as Council Orchestrator<BR>(Orchestrator MCP Server)
    participant C as Cortex<BR>(Cortex MCP Server)
    participant G as Guardian Cache<BR>(Cortex MCP - CAG)
    participant F as Forge<BR>(Forge MCP Server)

    Note over O: 1. Gap Analysis & Research
    O->>O: Identify Strategic Gap
    O->>O: Conduct Research (Intelligence Forge)
    O->>O: Generate Research Report

    Note over O, C: 2. Knowledge Ingestion (RAG Update)
    Note right of O: Operation: cortex_ingest_incremental()
    O->>C: ingest_incremental(report)
    C-->>O: Ingestion Complete (Chunks Created)

    Note over O, G: 3. Cache Synthesis (CAG Update)
    Note right of O: Operation: cortex_guardian_wakeup()
    O->>G: guardian_wakeup()
    G->>C: Query High-Priority Context
    C-->>G: Return Context
    G->>G: Update Hot Cache
    G-->>O: Cache Warm & Ready

    Note over O: Regular Cycle Complete

    rect rgb(255, 250, 205)
        Note over O, F: 4. Periodic Fine-Tuning (Manual/Scheduled)
        Note right of F: Operation: forge_fine_tune()<br/>Triggered manually or on major milestones
        O->>F: Request Adaptation Packet
        F->>C: Query Recent Learnings (cortex_query)
        C-->>F: Return Documents
        F->>F: Synthesize Training Data (forge_create_dataset)
        F-->>O: Dataset Ready
        Note over F: Human reviews dataset,<br/>runs fine_tune.py,<br/>deploys new Constitutional Mind
    end
```