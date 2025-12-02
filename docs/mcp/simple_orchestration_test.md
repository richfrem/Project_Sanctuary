# Simple Orchestration Test Case

**Purpose:** Validate the orchestration pattern in Claude Desktop before implementing in Council MCP.

**Test Case:** "Review Protocol 101 for compliance issues"

---

## Orchestration Flow

### Step 1: Get Context from Cortex (RAG)

**MCP Server:** `cortex` (mcp5)  
**Operation:** `cortex_query`  
**Command for Claude:**

```
mcp5_cortex_query(
  query="Protocol 101 Functional Coherence",
  max_results=3
)
```

**Expected Output:**
- 3 documents about Protocol 101
- Content includes protocol text and related context

**Save for next step:** Copy the first result's content

---

### Step 2: Dispatch to Auditor Persona

**MCP Server:** `agent_persona` (mcp1)  
**Operation:** `persona_dispatch`  
**Command for Claude:**

```
mcp1_persona_dispatch(
  role="auditor",
  task="Review Protocol 101 (Functional Coherence) and identify any compliance issues or areas that need clarification.",
  context={"protocol_content": "<paste content from Step 1>"},
  model_name="Sanctuary-Qwen2-7B:latest",
  engine="ollama"
)
```

**Expected Output:**
- Audit report from Sanctuary model
- Identified issues (if any)
- Takes ~30-60s

**Save for next step:** Copy the auditor's response

---

### Step 3: Dispatch to Strategist Persona

**MCP Server:** `agent_persona` (mcp1)  
**Operation:** `persona_dispatch`  
**Command for Claude:**

```
mcp1_persona_dispatch(
  role="strategist",
  task="Based on the auditor's findings, assess the strategic impact and recommend next steps.",
  context={"auditor_report": "<paste response from Step 2>"},
  model_name="Sanctuary-Qwen2-7B:latest",
  engine="ollama"
)
```

**Expected Output:**
- Strategic assessment
- Recommendations
- Takes ~30-60s

---

## Success Criteria

✅ **Step 1:** Cortex returns relevant documents  
✅ **Step 2:** Auditor analyzes with context  
✅ **Step 3:** Strategist builds on auditor's findings  

**Total Time:** ~60-120s (mostly LLM inference)

---

## What the Council Orchestrator Should Do

The `council_ops.py` `dispatch_task()` method should replicate this exact flow:

```python
def dispatch_task(self, task_description: str, agent: str = None, max_rounds: int = 3):
    # Step 1: Query Cortex
    rag_results = self.cortex.query(task_description, max_results=3)
    context = self._format_rag_context(rag_results)
    
    # Step 2: Dispatch to Auditor
    auditor_response = self.persona_ops.dispatch(
        role="auditor",
        task=task_description,
        context=context,
        model_name="Sanctuary-Qwen2-7B:latest"
    )
    
    # Step 3: Dispatch to Strategist (with auditor's findings)
    strategist_response = self.persona_ops.dispatch(
        role="strategist",
        task=f"Based on the auditor's findings, assess strategic impact: {auditor_response['response']}",
        context=context,
        model_name="Sanctuary-Qwen2-7B:latest"
    )
    
    return strategist_response
```

---

## Next Steps

1. **Execute this flow manually in Claude Desktop**
2. **Document the actual outputs** (save to a file for reference)
3. **Verify timing and quality**
4. **Implement the same logic in `council_ops.py`**
5. **Test that Council MCP produces identical results**
