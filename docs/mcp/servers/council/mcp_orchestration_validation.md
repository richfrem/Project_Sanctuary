# MCP Orchestration Validation Workflow

This document provides a **step-by-step validation workflow** that can be executed in Claude Desktop to verify the full orchestration pattern before implementing it in the Council MCP.

## Purpose

Validate each MCP operation individually to ensure all components work correctly before integrating them into the Council orchestrator. This follows the principle: **test components in isolation before composition**.

---

## Workflow: Protocol Review with Multi-Agent Analysis

**Scenario:** Review Protocol 110 (Agency and Sovereignty) for compliance with the Security Mandate.

### Step 1: Query Cortex for Context

**Tool:** `mcp5_cortex_query`

**Prompt for Claude:**
```
Please query the Cortex MCP for context about Protocol 110 and the Security Mandate:
- query: "Protocol 110 Agency and Sovereignty Security Mandate compliance"
- max_results: 3
```

**Expected Output:**
- 3 relevant documents from the knowledge base
- Protocol 110 content
- Security Mandate references

**Validation:**
- ✅ Documents retrieved
- ✅ Content is relevant
- ✅ Query time < 5s

---

### Step 2: Dispatch to Auditor for Review

**Tool:** `mcp1_persona_dispatch`

**Prompt for Claude:**
```
Using the context from Step 1, dispatch to the auditor persona:
- role: "auditor"
- task: "Review Protocol 110 (Agency and Sovereignty) for compliance with the Security Mandate. Identify any conflicts or areas of concern."
- context: {<paste the relevant content from Step 1>}
- model_name: "Sanctuary-Qwen2-7B:latest"
- engine: "ollama"
```

**Expected Output:**
- Detailed audit report
- Identified compliance issues (if any)
- Recommendations

**Validation:**
- ✅ Response received (expect ~30-60s)
- ✅ Response is contextually aware
- ✅ Audit quality is thorough

---

### Step 3: Dispatch to Strategist for Risk Assessment

**Tool:** `mcp1_persona_dispatch`

**Prompt for Claude:**
```
Using the auditor's findings from Step 2, dispatch to the strategist:
- role: "strategist"
- task: "Assess the strategic risks of the compliance issues identified by the auditor. Propose mitigation strategies."
- context: {<paste auditor's response from Step 2>}
- model_name: "Sanctuary-Qwen2-7B:latest"
- engine: "ollama"
```

**Expected Output:**
- Risk assessment
- Mitigation strategies
- Strategic recommendations

**Validation:**
- ✅ Response builds on auditor's findings
- ✅ Strategic perspective is evident
- ✅ Recommendations are actionable

---

### Step 4: (Optional) Update Protocol

**Tool:** `mcp9_protocol_update`

**Prompt for Claude:**
```
If the strategist recommends changes, update Protocol 110:
- number: 110
- updates: {"content": "<updated protocol text>"}
- reason: "Compliance with Security Mandate per Council review"
```

**Expected Output:**
- Protocol updated successfully
- Version incremented

**Validation:**
- ✅ Protocol file updated
- ✅ Metadata preserved

---

## Workflow: Code Review with Context Retrieval

**Scenario:** Review a code file for best practices.

### Step 1: Read Code File

**Tool:** `mcp3_code_read`

**Prompt for Claude:**
```
Read the target code file:
- path: "mcp_servers/lib/agent_persona/agent_persona_ops.py"
```

**Expected Output:**
- Full file contents

---

### Step 2: Dispatch to Auditor for Code Review

**Tool:** `mcp1_persona_dispatch`

**Prompt for Claude:**
```
Dispatch to auditor for code review:
- role: "auditor"
- task: "Review this code for best practices, potential bugs, and security issues."
- context: {"code": "<paste code from Step 1>"}
- model_name: "Sanctuary-Qwen2-7B:latest"
```

**Expected Output:**
- Code review findings
- Identified issues
- Recommendations

---

### Step 3: (Optional) Write Updated Code

**Tool:** `mcp3_code_write`

**Prompt for Claude:**
```
If the auditor recommends changes, write the updated code:
- path: "mcp_servers/lib/agent_persona/agent_persona_ops.py"
- content: "<updated code>"
```

---

## Success Criteria

For the Council orchestrator to replicate this workflow, it must:

1. ✅ **Query Cortex** for relevant context
2. ✅ **Dispatch to Agent Persona** with context injection
3. ✅ **Chain multiple agent calls** (auditor → strategist)
4. ✅ **Execute side effects** (update protocol, write code) based on agent recommendations
5. ✅ **Handle timeouts gracefully** (local LLM inference can take 30-60s)

---

## Performance Benchmarks

Based on validation tests:

| Operation | Avg Time | Notes |
|-----------|----------|-------|
| `cortex_query` | ~2-5s | Depends on DB size |
| `persona_dispatch` (Sanctuary) | ~30-60s | Local inference |
| `persona_dispatch` (GPT-4) | ~5-10s | Cloud API |
| `code_read` | <1s | File I/O |
| `protocol_update` | <1s | File I/O |

**Total workflow time (3 agents):** ~90-180s for local Sanctuary model

---

## Next Steps

1. Execute this workflow manually in Claude Desktop
2. Document any failures or unexpected behavior
3. Use the validated sequence as the reference implementation for `council_ops.py`
4. Implement the Council orchestrator to replicate this exact flow
