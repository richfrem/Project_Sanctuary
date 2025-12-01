# Task 078: Implement Agent Persona MCP & Refactor Council Orchestrator

**Status:** in-progress
**Priority:** High
**Lead:** Antigravity
**Dependencies:** Task 077 (Council MCP) ✅ COMPLETE
**Related Documents:** 
- ADR 040: Agent Persona MCP Architecture
- ADR 039: MCP Server Separation of Concerns
- `council_orchestrator/orchestrator/council/`
- `mcp_servers/council/README.md`

**CRITICAL DEPENDENCY ORDER:**
1. **Phase 1: Agent Persona MCP** (MUST be completed first)
2. **Phase 2: Orchestrator Refactoring** (depends on Agent Persona MCP)

The orchestrator refactoring CANNOT proceed until Agent Persona MCP is implemented and tested, as the orchestrator will become a client of the Agent Persona MCP.

---

## 1. Objective

Evolve the Council architecture from monolithic (internal agents) to modular (Agent Persona MCP servers). Create a single, configurable **Agent Persona MCP** that can assume any council member role (Coordinator, Strategist, Auditor), and refactor the Council Orchestrator to act as an MCP client coordinator.

---

## 2. Deliverables

### Phase 1: Agent Persona MCP Implementation
1. **Agent Persona MCP Server** (`mcp_servers/agent_persona/`)
   - `server.py`: FastMCP server with persona tools
   - `lib/agent_persona/agent_ops.py`: Agent operations library
   - Reuse `PersonaAgent` class from `council_orchestrator/orchestrator/council/agent.py`

2. **Core Tools**
   - `persona_dispatch(role, task, context, maintain_state)`: Execute task with specific persona
   - `persona_list_roles()`: List available persona roles
   - `persona_get_state(role)`: Retrieve agent conversation state
   - `persona_reset_state(role)`: Clear agent conversation history

3. **Persona Configuration**
   - Copy persona seed files to `mcp_servers/agent_persona/personas/`
   - Support custom persona files via configuration

4. **Testing**
   - Unit tests for Agent Persona MCP
   - Integration tests with each role (coordinator, strategist, auditor)
   - State persistence tests

5. **Documentation**
   - `mcp_servers/agent_persona/README.md`
   - Usage examples for each role
   - State management guide

### Design Principle: Extensibility for Custom Personas

The Agent Persona MCP must support **custom personas beyond the core three** (Coordinator, Strategist, Auditor):

**Future Personas:**
- **Security Reviewer**: Security audit and vulnerability assessment
- **Performance Analyst**: Performance optimization and bottleneck identification
- **Documentation Specialist**: Documentation quality and completeness review
- **UX Evaluator**: User experience and accessibility assessment
- **Cost Optimizer**: Resource usage and cost efficiency analysis
- **Compliance Officer**: Regulatory compliance and policy adherence
- **Innovation Scout**: Emerging technology and innovation opportunities

**Implementation:**
- Persona definitions stored as files in `mcp_servers/agent_persona/personas/`
- Tool accepts any role name, loads corresponding persona file
- If persona file doesn't exist, return clear error with available roles
- Support user-provided custom persona files via configuration

**Example Custom Persona:**
```
# personas/security_reviewer.txt

You are a Security Reviewer for Project Sanctuary.

Your role is to:
- Identify security vulnerabilities and risks
- Review code for security best practices
- Assess authentication and authorization mechanisms
- Evaluate data protection and encryption
- Check for common security anti-patterns (SQL injection, XSS, etc.)

Be thorough, critical, and provide actionable recommendations.
```

**Tool Usage:**
```python
# Use built-in persona
result = persona_dispatch(role="security_reviewer", task="Review authentication flow")

# Use custom persona (future)
result = persona_dispatch(
    role="custom_role",
    persona_file="/path/to/custom_persona.txt",
    task="..."
)
```

---
### Phase 2: Council Orchestrator Refactoring
1. **Orchestrator as MCP Client**
   - Add MCP client library to orchestrator
   - Implement agent discovery/configuration
   - Replace internal `PersonaAgent` calls with MCP tool calls

2. **Dual Mode Support** (Transition Period)
   - Support both internal agents AND Agent Persona MCP
   - Configuration flag: `use_mcp_agents: true/false`
   - Fallback to internal agents if MCP unavailable

3. **Consensus Mechanism**
   - Define how to synthesize responses from multiple agents
   - Implement voting/weighting logic
   - Handle agent disagreements

4. **Error Handling**
   - Retry logic for failed MCP calls
   - Fallback strategies if agent unavailable
   - Timeout handling

5. **Testing**
   - Test orchestrator with Agent Persona MCP
   - Test dual mode (internal + MCP)
   - Test error scenarios (agent unavailable, timeout)

### Phase 3: Documentation Updates
1. **Architecture Documentation** (`docs/mcp/`)
   - Update MCP architecture diagrams
   - Add Agent Persona MCP to ecosystem map
   - Document orchestrator-as-client pattern

2. **Sequence Diagrams**
   - User → LLM → Council MCP → Orchestrator → Agent Persona MCP
   - Multi-agent deliberation flow
   - Error handling flows

3. **Migration Guide**
   - How to transition from internal to MCP agents
   - Configuration examples
   - Troubleshooting guide

4. **Update Existing Docs**
   - `council_orchestrator/README.md`: Add MCP client architecture
   - `mcp_servers/council/README.md`: Update with Agent Persona MCP integration
   - `docs/mcp/mcp_operations_inventory.md`: Add Agent Persona MCP

---

## 3. Acceptance Criteria

### Phase 1: Agent Persona MCP
- [ ] Agent Persona MCP server runs and connects
- [ ] Can dispatch tasks to coordinator, strategist, auditor roles
- [ ] State persistence works (conversation history maintained)
- [ ] All tests passing (unit + integration)
- [ ] README documentation complete

### Phase 2: Orchestrator Refactoring
- [ ] Orchestrator can call Agent Persona MCP successfully
- [ ] Dual mode works (internal + MCP agents)
- [ ] Consensus mechanism implemented
- [ ] Error handling robust (retries, fallbacks, timeouts)
- [ ] All tests passing

### Phase 3: Documentation
- [ ] Architecture diagrams updated with Agent Persona MCP
- [ ] Sequence diagrams show full flow
- [ ] Migration guide complete
- [ ] All MCP READMEs updated
- [ ] `mcp_operations_inventory.md` updated

---

## 4. Implementation Plan

### Phase 1: Agent Persona MCP (Estimated: 6-8 hours)

#### Step 1.1: Scaffold
- [ ] Create `mcp_servers/agent_persona/` directory structure
- [ ] Create `mcp_servers/lib/agent_persona/` for operations library
- [ ] Copy persona seed files to `mcp_servers/agent_persona/personas/`

#### Step 1.2: Core Implementation
- [ ] Implement `AgentPersonaOperations` class
  - [ ] Load persona from file
  - [ ] Initialize `PersonaAgent` instance
  - [ ] Manage agent state (load/save)
- [ ] Implement `server.py` with FastMCP tools
  - [ ] `persona_dispatch` tool
  - [ ] `persona_list_roles` tool
  - [ ] `persona_get_state` tool
  - [ ] `persona_reset_state` tool

#### Step 1.3: Testing
- [ ] Create test harness (`tests/mcp_servers/agent_persona/`)
- [ ] Unit tests for each tool
- [ ] Integration tests with real LLM engines
- [ ] State persistence tests

#### Step 1.4: Documentation
- [ ] Create comprehensive README
- [ ] Add usage examples for each role
- [ ] Document state management

### Phase 2: Orchestrator Refactoring (Estimated: 8-10 hours)

#### Step 2.1: MCP Client Integration
- [ ] Add MCP client library to orchestrator dependencies
- [ ] Implement agent discovery (config file or environment)
- [ ] Create `AgentMCPClient` wrapper class

#### Step 2.2: Dual Mode Implementation
- [ ] Add configuration flag: `use_mcp_agents`
- [ ] Implement agent factory (internal vs MCP)
- [ ] Test both modes

#### Step 2.3: Deliberation Logic
- [ ] Refactor deliberation to use agent factory
- [ ] Implement consensus mechanism
- [ ] Add synthesis logic

#### Step 2.4: Error Handling
- [ ] Implement retry logic
- [ ] Add fallback to internal agents
- [ ] Add timeout handling
- [ ] Comprehensive error logging

#### Step 2.5: Testing
- [ ] Test with Agent Persona MCP
- [ ] Test dual mode
- [ ] Test error scenarios
- [ ] Integration tests

### Phase 3: Documentation (Estimated: 4-6 hours)

#### Step 3.1: Architecture Diagrams
- [ ] Create/update MCP ecosystem diagram
- [ ] Add Agent Persona MCP to architecture
- [ ] Show orchestrator-as-client pattern

#### Step 3.2: Sequence Diagrams
- [ ] User → LLM → Council → Orchestrator → Agent Persona flow
- [ ] Multi-agent deliberation sequence
- [ ] Error handling sequences

#### Step 3.3: Documentation Updates
- [ ] Update `council_orchestrator/README.md`
- [ ] Update `mcp_servers/council/README.md`
- [ ] Update `docs/mcp/mcp_operations_inventory.md`
- [ ] Create migration guide

---

## 5. Technical Specifications

### Agent Persona MCP Tool Signatures

```python
@mcp.tool()
def persona_dispatch(
    role: str,  # "coordinator" | "strategist" | "auditor" | "security_reviewer" | custom
    task: str,
    context: dict | None = None,
    maintain_state: bool = True,
    engine: str | None = None,  # "gemini" | "openai" | "ollama"
    custom_persona_file: str | None = None  # Path to custom persona file
) -> dict:
    """
    Dispatch a task to a specific persona agent
    
    Args:
        role: Persona role (built-in or custom)
        task: Task for the agent
        context: Optional context (from Cortex, previous decisions)
        maintain_state: Whether to persist conversation history
        engine: AI engine to use
        custom_persona_file: Path to custom persona definition (optional)
    
    Returns:
        {
            "role": "coordinator",
            "response": "Agent's response",
            "reasoning_type": "strategy" | "analysis" | "proposal",
            "session_id": "session_123",
            "state_preserved": true
        }
    """

@mcp.tool()
def persona_list_roles() -> dict:
    """
    List all available persona roles (built-in + custom)
    
    Returns:
        {
            "built_in": ["coordinator", "strategist", "auditor"],
            "custom": ["security_reviewer", "performance_analyst"],
            "total": 5
        }
    """

@mcp.tool()
def persona_create_custom(
    role: str,
    persona_definition: str,
    description: str
) -> dict:
    """
    Create a new custom persona
    
    Args:
        role: Unique role identifier (e.g., "security_reviewer")
        persona_definition: Full persona instruction text
        description: Brief description of the role
    
    Returns:
        {
            "role": "security_reviewer",
            "file_path": "personas/security_reviewer.txt",
            "status": "created"
        }
    """
```

### Orchestrator MCP Client Pattern

```python
# council_orchestrator/orchestrator/app.py

from mcp import ClientSession

async def deliberate_with_mcp(task: str):
    # 1. Query context from Cortex MCP
    async with ClientSession(cortex_mcp) as cortex:
        context = await cortex.call_tool("cortex_query", {"query": task})
    
    # 2. Get coordinator's plan
    async with ClientSession(agent_persona_mcp) as agent:
        coord_response = await agent.call_tool("persona_dispatch", {
            "role": "coordinator",
            "task": task,
            "context": context
        })
    
    # 3. Get strategist's assessment
    async with ClientSession(agent_persona_mcp) as agent:
        strat_response = await agent.call_tool("persona_dispatch", {
            "role": "strategist",
            "task": task,
            "context": context
        })
    
    # 4. Get auditor's review
    async with ClientSession(agent_persona_mcp) as agent:
        audit_response = await agent.call_tool("persona_dispatch", {
            "role": "auditor",
            "task": task,
            "context": context
        })
    
    # 5. Synthesize consensus
    return synthesize_decision(coord_response, strat_response, audit_response)
```

---

## 6. Migration Path

**Phase 1 (Current):** Monolithic orchestrator with internal agents ✅

**Phase 2 (Next):** 
- Create Agent Persona MCP
- Test with Auditor role only
- Orchestrator supports dual mode

**Phase 3 (Future):**
- Migrate all agents to MCP
- Deprecate internal agent implementation
- Orchestrator becomes pure MCP client

**Phase 4 (Vision):**
- Multiple Agent Persona MCP instances (horizontal scaling)
- Load balancing across agent instances
- Agent marketplace (swap implementations)

---

## 7. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| MCP communication overhead | Latency increase | Benchmark and optimize, consider caching |
| Agent unavailability | Deliberation failure | Fallback to internal agents, retry logic |
| State management complexity | Bugs, data loss | Comprehensive testing, state versioning |
| Configuration complexity | Setup difficulty | Clear documentation, sensible defaults |
| Consensus mechanism design | Incorrect decisions | Thorough testing, human oversight |

---

## 8. Success Metrics

- [ ] Agent Persona MCP response time < 5s per agent
- [ ] Orchestrator can handle agent failures gracefully
- [ ] 100% test coverage for Agent Persona MCP
- [ ] Zero regressions in deliberation quality
- [ ] Documentation complete and clear

---

## 9. Future Enhancements

- [ ] **Custom persona support** (user-defined agents)
  - [ ] `persona_create_custom` tool for creating new personas
  - [ ] Persona marketplace/registry
  - [ ] Persona versioning and updates
- [ ] **Specialized personas** (beyond core three)
  - [ ] Security Reviewer persona
  - [ ] Performance Analyst persona
  - [ ] Documentation Specialist persona
  - [ ] UX Evaluator persona
  - [ ] Cost Optimizer persona
- [ ] Agent performance metrics and monitoring
- [ ] A/B testing different agent implementations
- [ ] Multi-language agent support (Rust, Go, etc.)
- [ ] Real-time agent collaboration (streaming responses)
- [ ] Agent learning and adaptation (fine-tuning based on feedback)

---

**Total Estimated Effort:** 18-24 hours

**Recommended Schedule:**
- Week 1: Phase 1 (Agent Persona MCP)
- Week 2: Phase 2 (Orchestrator Refactoring)
- Week 3: Phase 3 (Documentation)
