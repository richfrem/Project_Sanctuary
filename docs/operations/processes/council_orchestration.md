# Council Orchestration Workflows

This document outlines the standard workflows for using the **Council MCP** and **Orchestrator MCP** within the 15-Domain Architecture (ADR 092). It is structured progressively, starting from basic building blocks and moving to complex self-evolving loops.

**Related Documentation:**
- Standard Orchestration Workflows (Pending)
- MCP Architecture & Testing (Pending)
- [Main Project README](../../../README.md) (Architecture Diagram)

---

## The Hierarchy of Orchestration

The system operates on three levels of abstraction:
1.  **Agent Persona MCP:** The "Neuron". Individual implementation of a persona (Auditor, Strategist).
2.  **Council MCP:** The "Brain". Orchestrates multi-agent deliberation and consensus.
3.  **Orchestrator MCP:** The "Will". Executes high-level missions and manages the Strategic Crucible Loop.

---

## Level 1: Basic Agent Dispatch (The Building Block)

**Concept:** Direct interaction with a single specific agent. The Council acts as a simple router to the Agent Persona MCP.

**Flow:**
`Council MCP` -> `Agent Persona MCP` -> `Forge LLM MCP`

**Common Use Case:**
- "I just need the Auditor to check this file."
- "I need the Strategist's opinion on this risk."

**Tool:** `council_dispatch`
- `agent`: "auditor" (or "strategist", "coordinator")
- `task_description`: Specific instruction

**Example:**
> "Audit 'docs/architecture/mcp/README.md' for broken links."

---

## Level 2: Context-Aware Deliberation

**Concept:** The Council retrieves relevant context from the **RAG Cortex MCP** before dispatching to an agent. This grounds the agent's response in Project Sanctuary protocols.

**Flow:**
`Council MCP` -> `RAG Cortex MCP (Query)` -> `Agent Persona MCP` -> `Forge LLM MCP`

**Common Use Case:**
- "How does this new feature align with Protocol 101?" (Requires knowing Protocol 101)

**Tool:** `council_dispatch`
- `agent`: "auditor"
- `task_description`: "Check compliance with Protocol 101"

**Internal Process:**
1. Council detects need for context.
2. Calls `cortex_query("Protocol 101")`.
3. Injects retrieved content into the Agent's context window.

---

## Level 3: Multi-Agent Consensus (Full Council)

**Concept:** The core "Council" capability. Multiple agents deliberate, critique one another, and reach a synthesized consensus.

**Flow:**
`Council MCP` -> `[Coordinator, Strategist, Auditor]` -> `Deliberation Logic` -> `Consensus`

**Common Use Case:**
- Complex architectural decisions.
- Risk assessments requiring multiple viewpoints.

**Tool:** `council_dispatch`
- `agent`: `None` (Defaults to full council)
- `max_rounds`: 2 or 3

**Internal Process:**
1. **Round 1:** All agents provide initial analysis.
2. **Round 2:** Agents critique Round 1 outputs.
3. **Synthesis:** Coordinator creates a final consensus output.

---

## Level 4: The Strategic Crucible Loop (Task 056)

**Concept:** The highest level of orchestration. The **Orchestrator MCP** manages the Council to identify gaps, creates solutions, and then **self-corrects** by updating the knowledge base (RAG).

**Flow:**
1. `Orchestrator` -> `Council` (Identify Gap)
2. `Orchestrator` -> `Code MCP` (Write Research/Fix)
3. `Orchestrator` -> `Git MCP` (Commit)
4. `Orchestrator` -> `RAG Cortex MCP` (Ingest = Learn)

**Common Use Case:**
- "Harden the self-evolving loop."
- "Research and document Protocol 116."

**Tool:** `orchestrator_run_strategic_cycle`
- `gap_description`: "Missing documentation for..."
- `research_report_path`: "00_CHRONICLE/..."

**Reference Diagrams:**
- [Continuous Learning Pipeline](../../../README.md#3-continuous-learning-pipeline) (Full Loop Interaction)
- [MCP Architecture Diagram](../../../README.md#mcp-architecture-diagram) (System Components)

---

## Summary of Tools

| Level | Tool | MCP Server | Purpose |
| :--- | :--- | :--- | :--- |
| **1** | `council_dispatch(agent="name")` | Council | Single Agent Router |
| **2** | `council_dispatch(agent="name")` | Council | RAG-Augmented Agent |
| **3** | `council_dispatch(agent=None)` | Council | Multi-Agent Consensus |
| **4** | `orchestrator_run_strategic_cycle` | Orchestrator | Full Self-Learning Loop |

---

## Validation Scenarios (Task 087)

To ensure the integrity of these levels, the following standardized test scenarios are tracked in **Task 087**:

### Level 1 Tests (Agent Chains)
Verify the `Council` -> `Agent` link.
- **Auditor Chain:** `council_dispatch(agent="auditor", ...)`
- **Strategist Chain:** `council_dispatch(agent="strategist", ...)`
- **Coordinator Chain:** `council_dispatch(agent="coordinator", ...)`

### Level 2 & 3 Tests (Orchestrator Chains)
Verify the `Orchestrator` -> `External MCP` links.
- **Council Chain:** `orchestrator_dispatch(mission="...")` (Calls Council)
- **Cortex Query Chain:** `orchestrator_dispatch` calling `cortex_query`
- **Cortex Ingest Chain:** `orchestrator_dispatch` calling `cortex_ingest_incremental`
- **Protocol Update Chain:** `orchestrator_dispatch` calling `protocol_update`

These scenarios provide the bottom-up verification required before running full Strategic Crucible Loops.
