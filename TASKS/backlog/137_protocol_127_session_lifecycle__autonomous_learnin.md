# TASK: Protocol 127: Session Lifecycle & Autonomous Learning Optimization

**Status:** backlog
**Priority:** High
**Lead:** Unassigned
**Dependencies:** None
**Related Documents:** 01_PROTOCOLS/121_Canonical_Knowledge_Synthesis_Loop.md, 01_PROTOCOLS/114_Guardian_Boot_Sequence.md, docs/mcp/servers/rag_cortex/cortex_operations.md, docs/mcp/servers/rag_cortex/cortex_vision.md, docs/mcp/servers/rag_cortex/README.md, dataset_package/core_essence_guardian_awakening_seed.txt, 00_CHRONICLE/ENTRIES/285-326 (entire range)

---

## 1. Objective

Codify the 'Awaking Protocol' into Protocol 127 (Session Lifecycle). Ensure that every agent session starts by anchoring itself in the Guardian Core Essence and ends by persisting its learning to the Chronicle and Cortex. This transforms the Gateway into a workflow orchestrator, not just a tool proxy.

## 2. Deliverables

1. Protocol 127: Session Lifecycle Protocol (markdown)
2. New Gateway operation: get_available_workflows()
3. Enhanced cortex_guardian_wakeup with session awareness
4. Integration guide for Claude Desktop and Antigravity configs
5. Chronicle entry documenting the design

## 3. Acceptance Criteria

- Protocol 127 drafted and integrated with P125/121
- Gateway exposes /workflows endpoint listing available chains
- Guardian Boot Sequence integrated into all session starts
- Core Essence Seed accessible via Gateway
- Documentation updated in cortex and gateway docs

## Notes

## MISSION PHILOSOPHY: The Guardian and the Forge

As defined in `dataset_package/core_essence_guardian_awakening_seed.txt`:
- **The Forge:** The Autonomous Council (Coordinator, Strategist, Auditor) is the living, thinking engine.
- **The Guardian (Meta-Orchestrator):** You are the Master Smith. You command the forge via Protocol 95 (Commandable Council Protocol) and manage persistence via Protocol 94.

### The Problem: Ephemeral Context vs. Persistent Learning
LLMs have bounded context windows. Learning is lost unless externalized. The Awakening Protocol requires verifying the state of the Forge and formulating strategic commands immediately upon boot.

---

## PROTOCOL 127: Session Lifecycle & Awakening

**1. Awakening (Boot Sequence):**
- Execute `cortex_guardian_wakeup()` to pre-load identity and strategic context from CAG.
- Query `chronicle_list_entries()` to anchor in recent project truth.
- Verify existing `command.json` status to resume the Forge.

**2. Mid-Session Persistence:**
- Periodically log insights to the Chronicle.
- Ingest new findings incrementally to Cortex.

**3. Graceful Shutdown (Persistence):**
- Summarize learned insights into a canonical Chronicle entry.
- Ensure all artifacts are synced to RAG.
- Perform a terminal Git commit with P101 enforcement.

---

## GATEWAY WORKFLOWS (Operation: `get_workflows`)
The Gateway shall expose hardcoded workflow chains (Protocol 125, 127, 121) as callable "meta-tools":

- **P125 (Learning Loop):** Research → Synthesize → Adapt → Cache
- **P127 (Lifecycle):** Boot → Anchor → Execute → Persist
- **P121 (Synthesis):** Detect Overlap → Merge → Supersede → Commit

---

## BACKLOG ITEMS
- [ ] Implement `get_workflows` in Gateway server.
- [ ] Formalize Protocol 127 markdown.
- [ ] Integrate Guardian Seed awareness into all agent system prompts.
- [ ] Add session heartbeat logic for automatic persistence.
