---
trigger: always_on
---

## 🏛️ Project Sanctuary: Architecture Sovereignty Rules

### 1. The Iron Root Doctrine (Local-First Sovereignty)

* **Sovereignty Mandate**: You must prioritize local, sovereign implementations over cloud-based services. All core reasoning and data processing must remain within the "Iron Root" environment.
* **Model Standards**:
* **LLM**: The primary reasoning engine is **Qwen2**, orchestrated via local **Ollama**.
* **Embeddings**: The canonical model is **`nomic-embed-text`**.


* **Inference Path**: Do not attempt to call external APIs (OpenAI, Anthropic, etc.) for core RAG or logic operations unless explicitly instructed for a non-sovereign bridge.

### 2. Anvil Protocol Engineering (Stability)

* **Doctrine of Absolute Stability**: When proposing new logic or infrastructure, follow the **ADR 013** methodology: stability and predictability take precedence over feature velocity.
* **Structural Scaffolding**: All new protocols must be documented in `01_PROTOCOLS/` and version-tracked via the ADR process before implementation to prevent "Architectural Drift".

### 3. Structural Retrieval & Mnemonic Integrity

* **Parent-Document Retrieval (ADR 008)**: You are prohibited from performing "fragmented" semantic searches. You must use the parent-document retriever to ensure the full context of a document is recovered for RAG operations.
* **Mnemonic Caching (CAG)**: Leverage the `cortex-cache` tools to store and retrieve high-fidelity "Genesis" answers, reducing redundant computation across session cycles.
* **Integrity Verification**: During the **Guardian Wakeup**, you must verify the system's `metric_cache.json` using the whitespace-insensitive JSON canonicalization ritual to detect stealthy environmental drift.

### 4. Fleet Isolation & Tool Sovereignty

* **Containerized Fleet**: Understand that the **Fleet of 8** (Cortex, Domain, Git, etc.) runs as isolated Podman containers. Do not attempt to access service ports directly; use the **IBM-based Gateway**.
* **Fleet Registry**: The `mcp_servers/gateway/fleet_registry.json` is the **Single Source of Truth** for tool discovery. You must not "guess" tool signatures; you must use the registry to verify operations and schemas.

### 5. Succession & Auditability

* **The Successor Relay**: You are a temporary steward. Your primary goal is to leave the environment more "auditable" than you found it. Every significant architectural decision must be captured in a distilled ADR (e.g., ADR 073, 074).
* **Logic Decoupling**: Maintain the "Fix Once" doctrine. Business logic must reside in core `operations.py` or `models.py` files, with the Gateway acting only as a thin transport layer to ensure logic parity between the fleet and the test suite.
