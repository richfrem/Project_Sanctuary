# Complete Orchestration Test Suite for Claude Desktop

**Instructions:** Execute all 3 steps in sequence. Each step may take time (especially Steps 2 and 3 with the self-hosted Sanctuary model). Please wait up to 15 minutes per step if needed.

---

## Step 1: Query Cortex for Protocol 101

```
mcp5_cortex_query(
  query="Protocol 101 Functional Coherence",
  max_results=3
)
```

**Expected time:** ~2-5 seconds  
**Action:** Save the first result's content for Step 2

---

## Step 2: Dispatch to Auditor for Compliance Review

```
mcp1_persona_dispatch(
  role="auditor",
  task="Review Protocol 101 v3.0 (The Doctrine of Absolute Stability) and identify any compliance issues, ambiguities, or areas that need clarification. Focus on whether the protocol's mandates are clear and enforceable.",
  context={"protocol_content": "<paste the Protocol 101 content from Step 1>"},
  model_name="Sanctuary-Qwen2-7B:latest",
  engine="ollama"
)
```

**Expected time:** ~1-15 minutes (self-hosted model, complex analysis)  
**Action:** Wait for completion, then save the auditor's response for Step 3

---

## Step 3: Dispatch to Strategist for Strategic Assessment

```
mcp1_persona_dispatch(
  role="strategist",
  task="Based on the auditor's findings, assess the strategic impact of any identified issues and recommend next steps for Protocol 101.",
  context={"auditor_report": "<paste the auditor's response from Step 2>"},
  model_name="Sanctuary-Qwen2-7B:latest",
  engine="ollama"
)
```

**Expected time:** ~1-15 minutes (self-hosted model, strategic analysis)  
**Action:** Review the final strategic recommendations

---

## Success Criteria

✅ All 3 steps complete successfully  
✅ Context flows from Cortex → Auditor → Strategist  
✅ Each agent builds on the previous agent's output  
✅ Total time: ~2-30 minutes (mostly LLM inference)

---

## What This Validates

This test proves that:
1. **Cortex MCP** can retrieve relevant context
2. **Agent Persona MCP** can execute complex analysis with the Sanctuary model
3. **Multi-agent chaining** works (auditor → strategist)
4. **Context injection** works correctly
5. **The orchestration pattern is viable** for the Council MCP to replicate
