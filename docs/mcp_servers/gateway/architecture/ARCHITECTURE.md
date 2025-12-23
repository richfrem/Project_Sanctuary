# MCP Gateway Architecture Specification

**Version:** 2.2 (Fleet Management)  
**Status:** Canonical  
**Last Updated:** 2025-12-20  
**References:** ADR 058, ADR 060, ADR 064, ADR 071

---

## 1. Overview

This document defines the technical architecture for the **Sanctuary MCP Gateway**, a centralized external broker that unifies 14+ MCP servers into a single endpoint for Claude Desktop.

**Core Philosophy:**
1.  **Externalization (ADR 058):** The Gateway runs as a "Black Box" service via Podman, decoupled from the main repo.
2.  **Hybrid Fleet (ADR 060):** 10 script-based servers are consolidated into a **Fleet of 8 Physical Containers** (6 logical clusters).

---

## 2. System Architecture

### 2.1 Fleet of 8 Architecture

The architecture consolidates individual tools into risk-based clusters to prevent orchestration fatigue while maintaining security boundaries.

```mermaid
---
config:
  theme: base
  layout: dagre
---
flowchart TB
    Client["<b>MCP Client</b><br>(Claude Desktop)"] -- HTTPS<br>(API Token Auth) --> Gateway["<b>Sanctuary MCP Gateway</b><br>External Service (Podman)<br>localhost:4444"]
    
    Gateway -- Docker Network --> Utils["<b>1. sanctuary_utils</b><br>:8100"]
    Gateway -- Docker Network --> Filesystem["<b>2. sanctuary_filesystem</b><br>:8101"]
    Gateway -- Docker Network --> Network["<b>3. sanctuary_network</b><br>:8102"]
    Gateway -- Docker Network --> Git["<b>4. sanctuary_git</b><br>:8103"]
    Gateway -- Docker Network --> Domain["<b>5. sanctuary_domain</b><br>:8105"]
    Gateway -- Docker Network --> Cortex["<b>6. sanctuary_cortex</b><br>:8104"]
    
    subgraph Backends["<b>Physical Fleet (Backends)</b>"]
        VectorDB["<b>7. sanctuary_vector_db</b><br>:8110"]
        Ollama["<b>8. sanctuary_ollama_mcp</b><br>:11434"]
    end

    Cortex --> VectorDB
    Cortex --> Ollama
    Domain --> Utils
    Domain --> Filesystem

    style Client fill:#e1f5ff,stroke:#0d47a1
    style Gateway fill:#fff4e1,stroke:#e65100
    style Utils fill:#e8f5e9,stroke:#2e7d32
    style Filesystem fill:#fff3e0,stroke:#ef6c00
    style Network fill:#f3e5f5,stroke:#7b1fa2
    style Git fill:#ffebee,stroke:#c62828
    style Domain fill:#e3f2fd,stroke:#1565c0
    style Cortex fill:#e0f2f1,stroke:#00695c
    style Backends fill:#f5f5f5,stroke:#9e9e9e
```

### 2.2 Fleet Management (Registration & Discovery)

The management of the Fleet follows a **3-Layer Declarative Pattern**, decoupling design intent from transport and runtime observation. This ensures the system remains resilient even if specific clusters are temporarily unreachable.

```mermaid
---
config:
  theme: base
  layout: dagre
---
flowchart LR
    subgraph INTENT["<b>Spec Layer</b> (fleet_spec.py)"]
        FLEET_SPEC["<b>FLEET_SPEC</b><br><i>Design Intent</i><br>• Slugs<br>• Default URLs"]
    end
    subgraph POLICY["<b>Resolver Layer</b> (fleet_resolver.py)"]
        RESOLVER["<b>Fleet Resolver</b><br><i>Policy Logic</i><br>• Env Overrides<br>• Docker Context"]
    end
    subgraph EXECUTION["<b>Execution Layer</b> (Transport)"]
        CLI["<b>CLI Orchestrator</b><br>(fleet_orchestrator.py)"]
        GATEWAY_CLIENT["<b>gateway_client.py</b><br><i>Pure Transport</i>"]
    end
    subgraph TESTING["<b>Testing Layer</b> (tests/...)"]
        TEST_CLIENT["<b>gateway_test_client.py</b>"]
        INTEG_TESTS["<b>Integration Tests</b><br>(clusters/...)"]
    end
    subgraph RUNTIME["<b>Runtime System</b>"]
        GATEWAY["<b>Sanctuary Gateway</b>"]
        MCP["<b>Fleet of MCP Servers</b>"]
    end
    subgraph OBSERVATION["<b>Observation Layer</b> (Non-Authoritative)"]
        REGISTRY_JSON["<b>fleet_registry.json</b><br><i>Discovery Manifest</i>"]
    end
    FLEET_SPEC -->|intent| RESOLVER
    RESOLVER -->|resolved endpoints| CLI
    RESOLVER -->|resolved endpoints| TEST_CLIENT
    
    CLI -->|invoke| GATEWAY_CLIENT
    TEST_CLIENT -->|wrap| GATEWAY_CLIENT
    TEST_CLIENT --> INTEG_TESTS
    
    GATEWAY_CLIENT -->|HTTP / SSE| GATEWAY
    GATEWAY --> MCP

    MCP -->|handshake| GATEWAY
    GATEWAY -->|observed tools| GATEWAY_CLIENT
    GATEWAY_CLIENT -->|write only| REGISTRY_JSON
    MCP -. unreachable .-> GATEWAY
    GATEWAY -. degraded state .-> REGISTRY_JSON
```

### 2.3 Component Responsibilities

#### The External Gateway (Broker)
- **Role:** Central entry point and router.
- **Location:** External repo (`sanctuary-gateway`), run via `podman`.
- **Function:** Authenticates clients, enforces allowlists, and routes tool calls to the appropriate Fleet container.
- **Security:** "Triple-Layer Defense" (Localhost-only, Bearer Token, Non-persistent).

#### The Fleet Clusters
1.  **sanctuary_utils**: Low-risk, stateless tools (Time, Calc, UUID, String).
2.  **sanctuary_filesystem**: High-risk file operations. Isolated from network.
3.  **sanctuary_network**: External web access (Brave, Fetch). Isolated from filesystem.
4.  **sanctuary_git**: Dual-permission (Filesystem + Network). Completely isolated container.
5.  **sanctuary-intelligence**:
    *   **Cortex (MCP):** The "Brain" that processes queries, manages **Cognitive Continuity (P128)**, and safeguards the **Guardian Role**.
    *   **VectorDB (Backend):** ChromaDB storage.
    *   **Ollama (Backend):** LLM inference.
6.  **sanctuary_domain**:
    *   **Role:** Hosts core Python business logic (Chronicle, Protocol, Task, ADR).
    *   **Port:** Exposes tools via SSE on port 8105.

---

## 3. Communication Protocols

### 3.1 Client to Gateway
- **Transport:** HTTPS (JSON-RPC 2.0)
- **Auth:** Standard `Authorization: Bearer <token>`
- **Endpoint:** `https://localhost:4444/sse`

### 3.2 Gateway to Fleet
- **Transport:** HTTP / SSE (Server-Sent Events)
- **Network:** Internal Docker/Podman network (`sanctuary-net`)
- **Discovery:** Dynamic Self-Registration (Containers POST their manifest to Gateway on startup).

---

## 4. Deployment Architecture

### 4.1 Podman Management
The entire system is orchestrated via `docker-compose.yml` (using Podman).

```yaml
services:
  # The Logical Clusters
  sanctuary_utils:
    image: sanctuary_utils:latest
    networks: [sanctuary-net]
  
  sanctuary_filesystem:
    image: sanctuary_filesystem:latest
    volumes: [./workspace:/app/workspace]
    networks: [sanctuary-net]

  # External Gateway (Managed separately, connects via network)
  # ...
```

### 4.2 Security Boundaries
- **Network Isolation:** Fleet containers do NOT expose ports to host (except for specific debugging). Only the Gateway exposes port 4444.
- **Volume Isolation:** Only `sanctuary_filesystem` and `sanctuary_git` have write access to the workspace.

---

## 5. Gateway-Routed Protocols

### 5.1 Recursive Learning Loop (P125)

The following diagram shows how the Learning Loop (Protocol 125) operates through the Gateway:

```mermaid
sequenceDiagram
    autonumber
    participant A as 🧠 Cognitive Agent<br>(Claude/Gemini)
    participant GW as 🌐 MCP Gateway<br>(Port 4444)
    participant Fleet as 🐳 Fleet of 8<br>(Podman)
    participant VDB as 📊 Vector DB
    participant LLM as 🤖 Ollama

    Note over A: Agent identifies learning opportunity
    
    rect rgb(230, 245, 255)
        Note over A, GW: 1. Tool Discovery
        A->>GW: GET /sse (Connect)
        GW-->>A: Available Tools (180+)
    end

    rect rgb(255, 245, 230)
        Note over A, Fleet: 2. Knowledge Ingestion
        A->>GW: cortex_ingest_incremental(doc)
        GW->>Fleet: Route to cortex:8104
        Fleet->>VDB: Embed → Store
        Fleet-->>GW: {doc_id}
        GW-->>A: Ingestion Complete
    end

    rect rgb(230, 255, 230)
        Note over A, LLM: 3. Semantic Verification (P125)
        A->>GW: cortex_query(topic)
        GW->>Fleet: Route to cortex:8104
        Fleet->>VDB: Similarity Search
        Fleet->>LLM: Augment Response
        Fleet-->>GW: {score: 0.94}
        GW-->>A: Echo-Back Verified
    end

    rect rgb(255, 230, 255)
        Note over A, Fleet: 4. Chronicle Entry
        A->>GW: chronicle_create_entry()
        GW->>Fleet: Route to domain:8105
        GW-->>A: Learning Loop Complete ✅
    end
```

### 5.2 Cognitive Continuity (P128)

Protocol 128 enforces a "Red Team Gate" and persistent identity via the **Guardian Role**.

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

---

## 6. References

- **ADR 058:** Decoupling Strategy (External Gateway)
- **ADR 060:** Hybrid Fleet Architecture (The 5 Clusters)
- **ADR 059:** JWT Authentication
- **ADR 062:** Rejection of n8n Automation (Manual Loop Reinforced)
- **ADR 071:** Protocol 128 (Cognitive Continuity)
