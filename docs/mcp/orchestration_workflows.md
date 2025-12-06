# Standard Council Orchestration Workflows

This document defines the standardized workflows for the Council Orchestrator to coordinate MCPs. These workflows are the building blocks for complex operations like the Strategic Crucible Loop.

**Related Documentation:**
- [MCP Architecture](../mcp/test_forge_mcp_and_RAG_mcp.md) (Full System Diagram)
- [Council Orchestration Levels](../../docs/workflows/council_orchestration.md) (Complexity Hierarchy)

---

## Workflow 1: Context Retrieval (Orchestrator -> Cortex)

**Purpose:** Retrieve relevant knowledge from the RAG Cortex before making decisions.
**Trigger:** Before any complex agent deliberation.

### Sequence Diagram

```mermaid
sequenceDiagram
    participant Orch as Orchestrator MCP
    participant Cortex as RAG Cortex MCP
    participant Chroma as ChromaDB

    Note over Orch: Need context for "Protocol 00"

    Orch->>Cortex: cortex_query("Protocol 00")
    Cortex->>Chroma: Vector Search
    Chroma-->>Cortex: Return Relevant Chunks
    Cortex-->>Orch: Return Markdown Content
```

---

## Workflow 2: Agent Deliberation (Orchestrator -> Council -> Agent)

**Purpose:** Delegate cognitive tasks to specialized personas.
**Trigger:** When a specific perspective (Audit, Strategy) is needed.

### Sequence Diagram

```mermaid
sequenceDiagram
    participant Orch as Orchestrator MCP
    participant Council as Council MCP
    participant Agent as Agent Persona MCP
    participant Forge as Forge LLM MCP

    Note over Orch: Need security review

    Orch->>Council: council_dispatch(agent="auditor", task="Review X")
    Council->>Agent: persona_dispatch(role="auditor", context=CTX)
    Agent->>Forge: query_model(prompt)
    Forge-->>Agent: Model Response
    Agent-->>Council: Auditor's Analysis
    Council-->>Orch: Final Deliberation Output
```

---

## Workflow 3: Action Execution (Orchestrator -> Code/Protocol/Git)

**Purpose:** Execute side effects based on Council recommendations.
**Trigger:** After a decision has been made.

### Sequence Diagram

```mermaid
sequenceDiagram
    participant Orch as Orchestrator MCP
    participant Code as Code MCP
    participant Protocol as Protocol MCP
    participant Git as Git MCP

    Note over Orch: Decision: Create Protocol 117

    par Parallel Execution
        Orch->>Protocol: protocol_create(117, "Content...")
        Orch->>Code: code_write("tests/test_p117.py", ...)
    end

    Orch->>Git: git_smart_commit("feat: add Protocol 117")
```

---

## Workflow 4: Multi-Agent Consensus (Council)

**Purpose:** Reach a consensus decision on complex topics.
**Trigger:** Strategic decisions or high-risk changes.

### Sequence Diagram

```mermaid
sequenceDiagram
    participant Orch as Orchestrator MCP
    participant Council as Council MCP
    participant Strategist as Agent (Strategist)
    participant Auditor as Agent (Auditor)
    participant Coordinator as Agent (Coordinator)

    Note over Orch: Decision: Open Source Strategy?

    Orch->>Council: council_dispatch(agent=None, "Debate Open Source")
    
    par Round 1
        Council->>Strategist: Analyze Risk
        Council->>Auditor: Check Compliance
    end
    
    Strategist-->>Council: High Risk / High Reward
    Auditor-->>Council: Need Licensing Review
    
    Note over Council: Internal Deliberation Logic
    
    Council->>Coordinator: Synthesize Consensus
    Coordinator-->>Council: Final Recommendation
    
    Council-->>Orch: Consensus Reached
```

---

## Workflow 5: Strategic Crucible Loop (Orchestrator self-correction)

**Purpose:** Identify gaps, research solutions, and update the knowledge base.
**Trigger:** `orchestrator_run_strategic_cycle` or automated schedule.

### Sequence Diagram

```mermaid
sequenceDiagram
    participant Orch as Orchestrator
    participant Council as Council
    participant Code as Code MCP
    participant Git as Git MCP
    participant Cortex as RAG Cortex

    Note over Orch: Step 1: Gap Analysis
    Orch->>Council: Identify Documentation Gaps
    Council-->>Orch: "Missing T081 Definitions"

    Note over Orch: Step 2: Research & Fix
    Orch->>Code: code_write("docs/mcp/orchestration_workflows.md")
    
    Note over Orch: Step 3: Commit
    Orch->>Git: git_smart_commit("docs: add workflows")
    
    Note over Orch: Step 4: Learn (Ingest)
    Orch->>Cortex: cortex_ingest_incremental(["docs/mcp/orchestration_workflows.md"])
    Cortex-->>Orch: Knowledge Updated
```

