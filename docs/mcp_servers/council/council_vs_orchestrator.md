# Council MCP vs Orchestrator MCP - Architectural Relationship

## Executive Summary

**Council MCP** and **Orchestrator MCP** are both orchestrators, but with different scopes:

- **Council MCP** = **Specialized Orchestrator** for multi-agent deliberation
- **Orchestrator MCP** = **General-Purpose Orchestrator** for strategic missions

## The Hierarchy

```
External LLM (Claude/Gemini/GPT)
    ↓
Orchestrator MCP (#10) - Strategic Mission Coordination
    ↓
Council MCP (#9) - Multi-Agent Deliberation Orchestration
    ↓
Agent Persona MCP (#8) - Individual Agent Execution
    ↓
LLM Engines (Ollama/OpenAI/Gemini)
```

---

## Council MCP (#9) - Specialized Orchestrator

### Purpose
**Specialized orchestrator** focused on **multi-agent deliberation workflows**

### Bounded Context
- **Single Responsibility:** Multi-agent deliberation
- **Orchestrates:** Multiple agent calls across deliberation rounds
- **Manages:** Agent conversation state, round tracking, consensus building

### Orchestration Scope: Tactical
```
Round 1: Coordinator → Strategist → Auditor
Round 2: Coordinator → Strategist → Auditor
Round 3: Synthesis and consensus
```

### Acts as Client To:
- **Agent Persona MCP** - Individual agent execution
- **Cortex MCP** - Knowledge base queries
- **Code MCP** - File operations (optional)
- **Git MCP** - Version control (optional)

### Example Operation
```python
council_dispatch(
    task_description="Review Protocol 101 for compliance issues",
    agent="auditor",
    max_rounds=3
)
```

**What happens internally:**
1. Council queries Cortex MCP for context about Protocol 101
2. Council calls Agent Persona MCP for "auditor" role (Round 1)
3. Council calls Agent Persona MCP for "strategist" role (Round 2)
4. Council calls Agent Persona MCP for "coordinator" role (Round 3)
5. Council synthesizes responses and returns deliberation result

### Performance Profile
- **Latency:** 30-60 seconds per agent call (LLM inference)
- **Bottleneck:** LLM inference via Agent Persona MCP
- **Scalability:** Can be scaled independently due to separation from Orchestrator

---

## Orchestrator MCP (#10) - General-Purpose Orchestrator

### Purpose
**General-purpose orchestrator** for **strategic mission coordination** across ALL MCPs

### Bounded Context
- **Broad Responsibility:** Strategic planning and multi-phase workflows
- **Orchestrates:** Complex missions involving many different MCPs
- **Manages:** Task lifecycle, mission state, cross-domain coordination

### Orchestration Scope: Strategic
```
Phase 1: Research (calls Cortex MCP, Council MCP)
Phase 2: Design (calls Council MCP, Protocol MCP)
Phase 3: Implement (calls Code MCP, Git MCP)
Phase 4: Verify (calls Council MCP, Task MCP)
Phase 5: Document (calls Chronicle MCP, ADR MCP)
```

### Acts as Client To:
- **Council MCP** - Multi-agent deliberation
- **Task MCP** - Task creation and tracking
- **Chronicle MCP** - Historical documentation
- **Protocol MCP** - Protocol management
- **Code MCP** - Code operations
- **Git MCP** - Version control
- **Cortex MCP** - Knowledge queries
- **ADR MCP** - Architecture decisions

### Example Operation
```python
orchestrator_dispatch_mission(
    mission="Implement Protocol 120 - MCP Composition Patterns",
    phases=["research", "design", "implement", "verify", "document"]
)
```

**What happens internally:**
1. **Phase 1 (Research):**
   - Orchestrator calls Cortex MCP to query existing patterns
   - Orchestrator calls **Council MCP** for strategic analysis
     - Council (in turn) calls Agent Persona MCP for agent execution
   
2. **Phase 2 (Design):**
   - Orchestrator calls **Council MCP** for design deliberation
   - Orchestrator calls Protocol MCP to create Protocol 120

3. **Phase 3 (Implement):**
   - Orchestrator calls Code MCP to write implementation
   - Orchestrator calls Git MCP to commit changes

4. **Phase 4 (Verify):**
   - Orchestrator calls **Council MCP** for review
   - Orchestrator calls Task MCP to track verification tasks

5. **Phase 5 (Document):**
   - Orchestrator calls Chronicle MCP to document completion
   - Orchestrator calls ADR MCP if architectural decisions were made

### Performance Profile
- **Latency:** Minutes to hours (multi-phase workflows)
- **Bottleneck:** Depends on phase (often Council MCP deliberations)
- **Scalability:** Coordinates long-running, complex workflows

---

## Key Differences

| Aspect | Council MCP | Orchestrator MCP |
|--------|-------------|------------------|
| **Purpose** | Multi-agent deliberation | Strategic mission coordination |
| **Scope** | Tactical (single deliberation) | Strategic (multi-phase missions) |
| **Orchestrates** | Agent calls in rounds | Entire workflows across MCPs |
| **Duration** | Minutes (2-5 minutes) | Minutes to hours |
| **Complexity** | Single-purpose, focused | Multi-purpose, broad |
| **Delegates To** | Agent Persona, Cortex | Council, Task, Chronicle, Protocol, Code, Git, etc. |

---

## Relationship: Delegation Pattern

**Orchestrator MCP delegates to Council MCP** when it needs multi-agent deliberation as part of a larger strategic workflow.

### Example: Implementing a New Protocol

```
External LLM: "Implement Protocol 120 - MCP Composition Patterns"
    ↓
Orchestrator MCP receives mission
    ↓
Phase 1: Research
    ↓
    Orchestrator calls Council MCP: "Analyze existing MCP composition patterns"
        ↓
        Council calls Agent Persona MCP (Coordinator)
        Council calls Agent Persona MCP (Strategist)
        Council calls Agent Persona MCP (Auditor)
        Council synthesizes analysis
        ↓
    Council returns analysis to Orchestrator
    ↓
Phase 2: Design
    ↓
    Orchestrator calls Council MCP: "Design Protocol 120 structure"
        ↓
        Council deliberates with agents
        ↓
    Council returns design to Orchestrator
    ↓
    Orchestrator calls Protocol MCP: "Create Protocol 120"
    ↓
Phase 3-5: Implement, Verify, Document
    (Orchestrator continues coordinating other MCPs)
```

---

## Testing Strategy

### 1. Test Agent Persona MCP in Isolation
```python
# Unit test: Verify individual agent execution works
persona_dispatch(
    role="auditor",
    task="Review test results"
)
```
**Verifies:** LLM integration, persona loading, response generation

---

### 2. Test Council MCP (Calls Agent Persona)
```python
# Integration test: Verify multi-agent deliberation works
council_dispatch(
    task_description="Strategic review of MCP architecture",
    agent="auditor",
    max_rounds=2
)
```
**Verifies:** 
- Council can call Agent Persona MCP
- Round tracking works
- Consensus synthesis works
- Cortex integration works

---

### 3. Test Orchestrator MCP (Calls Council)
```python
# End-to-end test: Verify strategic mission coordination works
orchestrator_dispatch_mission(
    mission="Complete system audit and documentation",
    phases=["research", "audit", "document"]
)
```
**Verifies:**
- Orchestrator can call Council MCP
- Multi-phase workflow management works
- Cross-MCP coordination works
- Task lifecycle management works

---

## Design Rationale (from ADR 042)

### Why Keep Them Separate?

1. **Single Responsibility Principle (SRP)**
   - Council: Deliberation orchestration only
   - Orchestrator: Mission orchestration only

2. **Scalability**
   - Council's LLM calls are the bottleneck (30-60s each)
   - Separating allows independent scaling of the slow layer

3. **Testability**
   - Clear test boundaries
   - Can test deliberation logic independently from mission logic

4. **Maintainability**
   - Changes to deliberation logic don't affect mission coordination
   - Changes to mission logic don't affect deliberation rounds

5. **Safety**
   - Smaller blast radius for failures
   - Easier to audit and verify each component

### Why Not Merge Them?

Merging would create a **monolithic orchestrator** that:
- ❌ Violates Single Responsibility Principle
- ❌ Creates performance bottlenecks (entire service blocked on LLM calls)
- ❌ Increases complexity and testing difficulty
- ❌ Reduces modularity and reusability
- ❌ Makes scaling harder

---

## Analogy: Company Organization

Think of it like a company:

- **Agent Persona MCP** = Individual employees (auditor, strategist, coordinator)
- **Council MCP** = Meeting room coordinator (orchestrates team discussions)
- **Orchestrator MCP** = Project manager (orchestrates entire projects using meetings, tasks, documentation)

When the project manager needs a team discussion, they delegate to the meeting room coordinator, who orchestrates the individual employees.

---

## Related Documentation

- **ADR 042:** [Separation of Council MCP and Agent Persona MCP](../../ADRs/042_separation_of_council_mcp_and_agent_persona_mcp.md)
- **ADR 039:** [MCP Server Separation of Concerns](../../ADRs/039_mcp_server_separation_of_concerns.md)
- **ADR 040:** [Agent Persona MCP Architecture](../../ADRs/040_agent_persona_mcp_architecture__modular_council_members.md)

---

**Status:** Architecture Documented ✅  
**Last Updated:** 2025-12-02
