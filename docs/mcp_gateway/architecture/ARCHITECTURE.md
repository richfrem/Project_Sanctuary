# MCP Gateway Architecture Specification

**Version:** 2.0 (Fleet of 7)  
**Status:** Canonical  
**Last Updated:** 2025-12-17  
**References:** ADR 058, ADR 060

---

## 1. Overview

This document defines the technical architecture for the **Sanctuary MCP Gateway**, a centralized external broker that unifies 12+ MCP servers into a single endpoint for Claude Desktop.

**Core Philosophy:**
1.  **Externalization (ADR 058):** The Gateway runs as a "Black Box" service via Podman, decoupled from the main repo.
2.  **Hybrid Fleet (ADR 060):** 10 script-based servers are consolidated into a **Fleet of 7 Physical Containers** (5 logical clusters).

---

## 2. System Architecture

### 2.1 Fleet of 7 Architecture

The architecture consolidates individual tools into risk-based clusters to prevent orchestration fatigue while maintaining security boundaries.

```mermaid
---
config:
  theme: base
  layout: dagre
---
flowchart TB
    Client["<b>MCP Client</b><br>(Claude Desktop)"] -- HTTPS<br>(API Token Auth) --> Gateway["<b>Sanctuary MCP Gateway</b><br>External Service (Podman)<br>localhost:4444"]
    
    Gateway -- Docker Network --> Utils["<b>1. sanctuary-utils</b><br>(Low Risk)<br>:8100/sse"]
    Gateway -- Docker Network --> Filesystem["<b>2. sanctuary-filesystem</b><br>(Privileged)<br>:8101/sse"]
    Gateway -- Docker Network --> Network["<b>3. sanctuary-network</b><br>(External)<br>:8102/sse"]
    Gateway -- Docker Network --> Git["<b>4. sanctuary-git</b><br>(Dual-Perm)<br>:8103/sse"]
    Gateway -- Docker Network --> Domain["<b>6. sanctuary-domain</b><br>(Business Logic)<br>:8105/sse"]
    Gateway -- Docker Network --> Cortex
    
    subgraph Intelligence["<b>5. Intelligence Cluster</b>"]
        Cortex["<b>5a. sanctuary-cortex</b><br>(MCP Server)<br>:8104/sse"]
        VectorDB["<b>5b. sanctuary-vector-db</b><br>(Backend)<br>:8000"]
        Ollama["<b>5c. sanctuary-ollama-mcp</b><br>(Backend)<br>:11434"]
        
        Cortex --> VectorDB
        Cortex --> Ollama
    end

    style Client fill:#e1f5ff,stroke:#0d47a1
    style Gateway fill:#fff4e1,stroke:#e65100
    style Utils fill:#e8f5e9,stroke:#2e7d32
    style Filesystem fill:#fff3e0,stroke:#ef6c00
    style Network fill:#f3e5f5,stroke:#7b1fa2
    style Git fill:#ffebee,stroke:#c62828
    style Intelligence fill:#e0f2f1,stroke:#00695c
    style Domain fill:#e3f2fd,stroke:#1565c0
```

### 2.2 Component Responsibilities

#### The External Gateway (Broker)
- **Role:** Central entry point and router.
- **Location:** External repo (`sanctuary-gateway`), run via `podman`.
- **Function:** Authenticates clients, enforces allowlists, and routes tool calls to the appropriate Fleet container.
- **Security:** "Triple-Layer Defense" (Localhost-only, Bearer Token, Non-persistent).

#### The Fleet Clusters
1.  **sanctuary-utils**: Low-risk, stateless tools (Time, Calc, UUID, String).
2.  **sanctuary-filesystem**: High-risk file operations. Isolated from network.
3.  **sanctuary-network**: External web access (Brave, Fetch). Isolated from filesystem.
4.  **sanctuary-git**: Dual-permission (Filesystem + Network). Completely isolated container.
5.  **sanctuary-intelligence**:
    *   **Cortex (MCP):** The "Brain" that processes queries.
    *   **VectorDB (Backend):** ChromaDB storage.
    *   **Ollama (Backend):** LLM inference.
6.  **sanctuary-domain**:
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
  sanctuary-utils:
    image: sanctuary-utils:latest
    networks: [sanctuary-net]
  
  sanctuary-filesystem:
    image: sanctuary-filesystem:latest
    volumes: [./workspace:/app/workspace]
    networks: [sanctuary-net]

  # External Gateway (Managed separately, connects via network)
  # ...
```

### 4.2 Security Boundaries
- **Network Isolation:** Fleet containers do NOT expose ports to host (except for specific debugging). Only the Gateway exposes port 4444.
- **Volume Isolation:** Only `sanctuary-filesystem` and `sanctuary-git` have write access to the workspace.

---

## 5. Gateway-Routed Learning Loop

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

---

## 6. References

- **ADR 058:** Decoupling Strategy (External Gateway)
- **ADR 060:** Hybrid Fleet Architecture (The 5 Clusters)
- **ADR 059:** JWT Authentication
- **ADR 062:** Rejection of n8n Automation (Manual Loop Reinforced)
