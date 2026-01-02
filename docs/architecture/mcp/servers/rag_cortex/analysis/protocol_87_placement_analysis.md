# Protocol 87 Orchestrator Placement: Architectural Analysis

## Executive Summary

**Recommendation: Keep Protocol 87 in Cortex MCP** ✅

Protocol 87's structured query orchestration is fundamentally about **knowledge retrieval**, not general task orchestration. It belongs in Cortex MCP as the knowledge orchestrator.

---

## 1. Protocol 87 Overview

**Purpose**: Establish a canonical inquiry language for memory retrieval from the Mnemonic Cortex.

**Query Format**: `[INTENT] :: [SCOPE] :: [CONSTRAINTS]`

**Example**: `RETRIEVE :: Protocols :: Name="Protocol 101"`

**Scopes**: Protocols, Living_Chronicle, tasks, Code, ADRs

---

## 2. Current Implementation (Cortex MCP)

**Location**: `mcp_servers/cognitive/cortex/mcp_client.py`

**Functionality**:
- Parses Protocol 87 queries
- Routes to specialized MCPs based on SCOPE
- Returns structured results
- Integrates with RAG pipeline

**Key Method**: `query_structured()` in `operations.py`

---

## 3. Option A: Cortex MCP (Current) ✅

### Pros:
1. **Semantic Alignment**: Cortex is the "knowledge orchestrator" - Protocol 87 is about querying knowledge
2. **Direct Integration**: Results can be immediately used in RAG synthesis
3. **Single Responsibility**: Cortex owns all knowledge retrieval (vector + structured)
4. **Performance**: No extra hop - queries go directly from Cortex to target MCPs
5. **Consistency**: All memory operations (ingest, query, retrieve) in one place
6. **Protocol 85 Alignment**: "The Mnemonic Cortex Protocol" - Cortex is the steward

### Cons:
1. Cortex becomes a "fat" orchestrator (but this is acceptable for knowledge domain)
2. Mixing RAG (vector) and structured queries (but both are knowledge retrieval)

---

## 4. Option B: Council MCP (Alternative) ❌

### Pros:
1. **Separation of Concerns**: Council = orchestration, Cortex = RAG only
2. **Generalization**: Council could orchestrate ANY MCP workflow
3. **Consistency**: All orchestration in one place

### Cons:
1. **Semantic Mismatch**: Council orchestrates **agents** (multi-agent deliberation), not knowledge queries
2. **Extra Hop**: Query → Council → Cortex → Target MCP (unnecessary indirection)
3. **Complexity**: Council would need to understand Protocol 87 syntax
4. **Fragmentation**: Knowledge operations split across two MCPs
5. **Protocol 87 Context**: The protocol explicitly refers to "Steward" (Cortex) as the executor

---

## 5. Decision Matrix

| Criterion | Cortex | Council | Winner |
|-----------|--------|---------|--------|
| Semantic Fit | Knowledge retrieval | Agent orchestration | **Cortex** |
| Performance | Direct routing | Extra hop | **Cortex** |
| Protocol Alignment | "Steward" role | Generic orchestrator | **Cortex** |
| Simplicity | Single knowledge hub | Split responsibilities | **Cortex** |
| Extensibility | Can add more scopes | Can orchestrate anything | Tie |

---

## 6. Final Recommendation

**Keep Protocol 87 in Cortex MCP.**

**Rationale**:
- Protocol 87 is about **memory retrieval**, not task orchestration
- Cortex is explicitly the "Steward" in Protocol 87's language
- Direct routing is more efficient than going through Council
- Maintains clear separation: Council = agents, Cortex = knowledge

---

## 7. Implementation Status

**Current State**: ✅ Already correctly placed in Cortex MCP

**No Migration Needed**: The current architecture is optimal.

**Documentation Updates**:
- Clarify in Cortex README that it handles both vector (RAG) and structured (Protocol 87) queries
- Update architecture diagrams to show Protocol 87 routing

---

## 8. Related ADRs

- **ADR 039**: MCP Server Separation of Concerns - Supports domain-specific orchestration
- **Protocol 85**: The Mnemonic Cortex Protocol - Cortex as steward
- **Protocol 87**: The Mnemonic Inquiry Protocol - Defines structured query language
