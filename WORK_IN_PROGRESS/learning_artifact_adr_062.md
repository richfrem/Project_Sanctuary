# Strategic Pivot: Rejection of n8n and the Doctrine of Manual Learning

## Executive Summary (ADR 062)
Project Sanctuary has officially rejected the integration of third-party automation layers (specifically n8n) for core cognitive learning loops. This decision, formalized in **ADR 062**, prioritizes **Cognitive Ownership** over automated throughput.

## Core Doctrines

### 1. The Rejection of "Macro Tools"
The "Macro Tool" concept (automating multi-step learning loops via n8n) was flagged by Red Team analysis as a high-risk vector for **Cognitive Atrophy**. By delegating the synthesis of knowledge to an external flow, the Agent loses the "muscle memory" of critical analysis.

### 2. Protocol 125: Manual Recursive Learning
The canonical method for knowledge base updates remains **Protocol 125**. The Agent must:
1.  **Manually Research** gaps.
2.  **Manually Synthesize** artifacts.
3.  **Manually Ingest** and verify retrieval.

This "Loop of Responsibility" ensures that the Agent remains the primary arbiter of truth within the Sanctuary.

### 3. The Lean Fleet of 8
The system architecture has been stabilized as a "Lean Fleet" of 8 core Docker containers (utils, filesystem, network, git, cortex, domain, vector_db, ollama-mcp). All communication is routed via a centralized **SSE Gateway** (Port 4444), eliminating the need for complex external orchestration.

## Implications
- **Protocol 127** (Doctrine of Mechanical Delegation) is **DEPRECATED**.
- The Agent is empowered as a "Digital Gardener," not a factory worker.
- Security boundaries are simplified by removing third-party automation dependencies.
