# TASK: Validate Lean Fleet & Execute Manual Learning Loop (Protocol 125)

**Status:** backlog
**Priority:** High
**Lead:** Unassigned
**Dependencies:** Fleet of 8 Deployment Complete
**Related Documents:** Protocol 125, Protocol 123, Protocol 121

---

## 1. Objective

Validate the operational **Lean Fleet of 8** and execute a complete manual learning loop to prove Agent-driven cognitive ownership.

> [!IMPORTANT]
> **Architectural Decision:** Protocol 127 (Mechanical Delegation) has been **REJECTED**. The Learning Loop remains fully manual per Protocol 125 to preserve Agent autonomy.

---

## 2. Lean Fleet Architecture (8 Containers)

| # | Container | Port | Role |
|---|-----------|------|------|
| 1 | sanctuary-utils | 8100 | Utility tools |
| 2 | sanctuary-filesystem | 8101 | File I/O |
| 3 | sanctuary-network | 8102 | HTTP fetching |
| 4 | sanctuary-git | 8103 | Git workflow |
| 5 | sanctuary-cortex | 8104 | RAG engine |
| 6 | sanctuary-domain | 8105 | Business logic |
| 7 | sanctuary-vector-db | 8000 | ChromaDB |
| 8 | sanctuary-ollama-mcp | 11434 | LLM inference |

**No Container #9 (n8n):** Background automation rejected.

---

## 3. Verification Phases

### Phase 1: Fleet Connectivity (Pulse Check)
- [ ] Verify all 8 containers running (`podman compose ps`)
- [ ] Health endpoints responding (8100-8105)
- [ ] VectorDB heartbeat (8000)
- [ ] Ollama model loaded (11434)

### Phase 2: RAG Cortex Qualification
- [ ] `cortex_ingest_incremental` - single document
- [ ] `cortex_query` - semantic search returns result
- [ ] Verification score > 0.90

### Phase 3: Manual Learning Loop (Protocol 125)
**The Agent executes each step manually:**
1. [ ] **Research:** Agent identifies knowledge gap
2. [ ] **Synthesize:** Agent creates learning artifact
3. [ ] **Ingest:** `cortex_ingest_incremental(artifact)`
4. [ ] **Verify:** `cortex_query(topic)` - confirm retrieval
5. [ ] **Chronicle:** `chronicle_create_entry(summary)`

> [!NOTE]
> **Cognitive Ownership:** Each step is explicitly controlled by the Agent. No macro tools, no delegation.

### Phase 4: Manual Gardener Routine
Since no background scheduler exists, verify manual maintenance via Gateway:
- [ ] `code_list_files("00_CHRONICLE", pattern="*.md")` - list docs
- [ ] Check for stale files (`last_verified` > 30 days)
- [ ] `network_fetch_url` - validate external links

---

## 4. Dual-Stack Verification Matrix

| Container | Local (STDIO) | Fleet (Docker) |
|-----------|---------------|----------------|
| utils | ✅ | ✅ |
| filesystem | ✅ | ✅ |
| network | ✅ | ✅ |
| git | ✅ | ✅ |
| cortex | ✅ | ✅ |
| domain | ✅ | ✅ |
| vector-db | ✅ | ✅ |
| ollama | ✅ | ✅ |

---

## 5. Acceptance Criteria

- [ ] All 8 containers operational (Fleet column = ✅)
- [ ] Manual Learning Loop completed end-to-end
- [ ] Semantic verification score > 0.90
- [ ] Chronicle entry documenting the loop

---

## Notes

**Protocol Status:**
- ❌ **Protocol 127 (Mechanical Delegation):** DEPRECATED - Risk of cognitive atrophy
- ✅ **Protocol 125 (Recursive Learning):** CANONICAL - Agent maintains full control
- ✅ **Protocol 123 (Signal/Noise):** Active - Rubric applied manually by Agent

**Status History:**
- 2025-12-19: Renamed to "Lean Fleet" - n8n/automation rejected
- 2025-12-19: todo → backlog (waiting for fleet deployment)
