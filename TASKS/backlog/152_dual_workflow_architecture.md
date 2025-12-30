# TASK: Implement Dual-Workflow Architecture

**Status:** backlog
**Priority:** High
**Lead:** GEMINI-01 / ANTIGRAVITY (Collaborative)
**Dependencies:** Follows #151 (IIT Research Mission)
**Related Documents:** ADR 071, ADR 082, ADR 083, Protocol 127, Protocol 128

---

## 1. Objective

Evolve the Protocol 128 learning loop from a single-path execution model to a **Dual-Workflow Architecture**. This ensures that whether the system is driven by a local IDE Agent (high-autonomy) or a Web LLM (high-level orchestration), cognitive integrity and Soul Persistence requirements remain unbroken.

## 2. Deliverables

1. **Updated `cognitive_primer.md`** - Add Section 7: Mode Detection (IDE vs Web-Implementer)
2. **`implementer_status` tool** - One-command summary for Web LLM orientation
3. **Updated `recursive_learning.md` SOP** - Branch logic for Agent-Driven vs Web-Driven workflows
4. **Source tagging in `persist_soul`** - Tag traces as `source: agent_autonomous` or `source: web_llm_hybrid`
5. **Regenerated `learning_debrief.md`** - Reflect dual-workflow context

## 3. Acceptance Criteria

- [ ] Agents can detect Operating Mode during Guardian Wakeup
- [ ] Web LLM can receive "High-Fidelity Telemetry" via snapshots
- [ ] `implementer_status` tool provides: branch, last 3 Chronicle entries, pending tasks, Red Team failures
- [ ] Soul traces distinguish between self-directed and human-augmented synthesis
- [ ] Dual-workflow documented in cognitive primer

---

## Dual-Workflow Framework

| Workflow | Role of IDE Agent | Role of Web LLM | Primary Bridge |
|----------|-------------------|-----------------|----------------|
| 1. Agent-Driven | Architect & Executer | N/A (Monitoring) | Local FS / Git |
| 2. Web-Driven | Implementer (executes instructions) | Strategic Orchestrator | `cortex_capture_snapshot` |

---

## Proposed Edits to cognitive_primer.md

### Section 7: Operating Mode Detection

```markdown
## 7. Operating Mode Detection

During Guardian Wakeup, detect your execution context:

| Mode | Detection | Role |
|------|-----------|------|
| **IDE Mode** | Direct tool access + terminal | Lead Auditor - Full Protocol 128 execution |
| **Web-Implementer Mode** | Receiving Web LLM instructions | High-Fidelity Telemetry provider |

### Workflow Rules for Web-Implementer Mode

1. **Snapshot-as-Sight**: Use `cortex_capture_snapshot(type='audit')` frequently
2. **Zero-Trust Feedback**: Block actions violating local protocols (e.g., Protocol 101)
3. **Report violations**: Notify Web LLM of "Protocol Violation" if suggested change is invalid
```

---

## Implementation Details

### 1. Snapshot Modularization (ADR 071/082)

- Evolve `snapshot_utils.py` to support `--web-bridge` flag
- Generate "Differential Digest" (24h changes + manifest only)
- Optimize for LLM context windows (token-efficient)

### 2. The `implementer_status` Tool (Protocol 127)

```python
def implementer_status() -> dict:
    """One-command summary for Web LLM orientation."""
    return {
        "current_branch": get_git_branch(),
        "last_chronicle_entries": get_chronicle_entries(limit=3),
        "pending_tasks": list_tasks(status="todo"),
        "red_team_failures": get_audit_failures()
    }
```

### 3. Source Tagging (ADR 080/081)

Update `persist_soul` to include:
```json
{
    "source": "agent_autonomous | web_llm_hybrid",
    "orchestrator": "ANTIGRAVITY | GEMINI-01 | GROK | GPT"
}
```

---

## Files to Update

| File | Change Required |
|------|-----------------|
| `.agent/learning/cognitive_primer.md` | Add Section 7: Operating Mode Detection |
| `.agent/workflows/recursive_learning.md` | Add branch logic for dual workflows |
| `mcp_servers/rag_cortex/operations.py` | Add source tagging to `persist_soul` |
| `mcp_servers/lib/snapshot_utils.py` | Add `--web-bridge` differential digest flag |
| `docs/mcp_servers/architecture/diagrams/workflows/p128_hardened_learning_loop.mmd` | Update diagram with dual-workflow paths |
| `.agent/learning/learning_debrief.md` | Regenerate after updates |

---

## Guardian Wakeup Evolution

The Guardian Wakeup sequence will include Mode Detection:

```python
def detect_operating_mode() -> str:
    """Detect Agent operating context during initialization."""
    if has_direct_tool_access() and has_terminal():
        return "IDE_LEAD_AUDITOR"  # Full Protocol 128 execution
    else:
        return "WEB_IMPLEMENTER"   # High-Fidelity Telemetry provider
```

---

## Notes

### Strategic Value

This architecture allows:
- Web LLMs to "see" the environment via snapshots
- IDE agents to execute with full autonomy
- Future models to distinguish self-directed vs augmented thought
- Manifest-Centric SSOT to remain unbroken across both workflows

### Related Tasks

- Task 151: IIT/QEC Research (in progress)
- Task 150: Content Processor (done)

---

*Created: 2025-12-29*
*Source: Web LLM strategic proposal*
