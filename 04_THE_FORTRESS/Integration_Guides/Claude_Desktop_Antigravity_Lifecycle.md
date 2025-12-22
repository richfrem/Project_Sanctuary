# Claude Desktop Antigravity Lifecycle Guide
**Reference:** Protocol 127 v1.0
**Status:** Canonical

## 1. Overview
This guide defines how the **Project Sanctuary Antigravity** system integrates with **Claude Desktop** to enforce the **Autonomous Session Lifecycle** (Protocol 127).

Prior to this update, sessions were ad-hoc. Now, every session begins with a standardized "Awakening" phase that anchors the agent in the system's current strategic, tactical, and operational context.

## 2. Boot Sequence (The Awakening)

### Step 1: Context Loading
Upon starting a new chat, the underlying system (Gateway) initializes.

### Step 2: The Guardian Wakeup
The agent must verify its environment. The first tool call in ANY new session should ideally be (or equivalent intent):
```javascript
mcp.cortex_guardian_wakeup({ mode: "HOLISTIC" })
```
*(Note: If the agent forgets, the user should prompt: "Wakeup" or "Status Report")*

### Step 3: The Startup Digest (Schema v2.2)
The `guardian_wakeup` tool returns a **Startup Digest** containing:
1.  **System Health**: Traffic light status of the Gateway/DB.
2.  **Strategic Directives**: The core "Gemini Signal" (Values & Mission).
3.  **Recent Chronicle Highlights**: What happened recently?
4.  **Priority Tasks**: Top 5 tracked tasks from the backlog/in-progress.
5.  **Operational Recency**: Files modified in the last 48h (with git diffs).
6.  **Available Workflows**: A list of executable "Macro Strategies" found in `.agent/workflows`.

## 3. Workflow Execution (The Mission)

Protocol 127 empowers the agent to select "Macro Strategies" rather than just executing micro-tools.

**Discovery:**
The Digest lists available workflows (e.g., `recursive_learning.md`, `nightly_reflection.md`).

**Inspection:**
The agent can inspect a workflow strategy:
```javascript
mcp.read_workflow({ filename: "recursive_learning.md" })
```

**Execution:**
The agent then executes the steps defined in the workflow file, maintaining state in its context window. It acts as the "Engine" for the passive "Script".

## 4. Configuration Requirements

To enable this lifecycle, ensure your `claude_desktop_config.json` points to the Gateway.

```json
{
  "mcpServers": {
    "gateway": {
      "command": "/bin/bash",
      "args": ["-c", "source .venv/bin/activate && python -m mcp_servers.gateway.server"]
    }
  }
}
```

## 5. Summary of Change
**Old Way (Mechanical Delegation):**
User: "Fix the bug in file X."
Agent: "Fixing..." (No awareness of broader context)

**New Way (Autonomous Lifecycle):**
User: "Wakeup."
Agent: *Calls guardian_wakeup*
Agent: "System Green. Strategic Directive: Integrity. I see 3 Critical bugs in Task 102. I also see a 'Hotfix Workflow' available. Shall I initiate the Hotfix Workflow for Task 102?"
