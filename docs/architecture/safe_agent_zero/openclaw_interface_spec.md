# OpenClaw Interface Specification (Safe Agent Zero)

**Status**: Draft
**Version**: 1.0

## 1. Overview

This specification defines how the "Sanctum" system interacts with **Agent Zero** (running OpenClaw). The interface is designed to maximize isolation while providing robust control to the user via the "Control Network".

## 2. Communication Interface: Agent Client Protocol (ACP)

We will use the **Agent Client Protocol (ACP)** to communicate with Agent Zero. This is OpenClaw's native protocol for IDEs and tools.

### 2.1 Transport
*   **Protocol**: WebSocket (Secure)
*   **Port**: `18789` (Default Gateway Port)
*   **Network**: `control-net` (Internal Docker Network)
*   **Binding**: `0.0.0.0` (Inside container, but only accessible to `frontend-net` via Docker networking).

### 2.2 Message Types (Command & Control)

The Nginx Guard (or a lightweight bridge service) will translate User Interface actions into ACP messages.

| Action | ACP Message Type | Payload Example | Description |
| :--- | :--- | :--- | :--- |
| **New Task** | `prompt` | `{"text": "Research OpenClaw architecture"}` | Sends a new instruction to the agent. |
| **Stop** | `cancel` | `{"runId": "active-run-id"}` | Immediately halts the current execution loop. |
| **Status** | `listSessions` | `{"limit": 10}` | Retrieves active/past sessions for the UI. |

## 3. Security Interface: Execution Approvals

To satisfy the "Human-in-the-Loop" (HITL) requirement from the Threat Model, we will leverage OpenClaw's native `ExecApprovals` system.

### 3.1 Configuration (`exec-approvals.json`)
We will mount a pre-configured approvals file into the container at `/home/node/.openclaw/exec-approvals.json`.

```json
{
  "version": 1,
  "defaults": {
    "security": "allowlist", 
    "ask": "always",
    "askFallback": "deny"
  },
  "agents": {
    "default": {
      "allowlist": [
        { "pattern": "ls -la" },
        { "pattern": "cat *" }
      ]
    }
  }
}
```

*   **`security`: "allowlist"**: Only commands explicitly matching a pattern are allowed automatically.
*   **`ask`: "always"**: Triggers an approval request for *every* command (or "on-miss" for unlisted ones).
*   **`askFallback`: "deny"**: If the user cannot be reached to approve, the command is blocked.

### 3.2 Authorization Table (Gated vs Autonomous)

| Action Category | Specific Action | Status | Approval Required? |
| :--- | :--- | :--- | :--- |
| **Reading (Safe)** | `browser.goto(url)` | **Autonomous** | ❌ No |
| | `browser.click(selector)` | **Autonomous** | ❌ No |
| | `fs.readFile(path)` | **Autonomous** | ❌ No (if in allowed dir) |
| | `http.get(url)` | **Autonomous** | ❌ No |
| **Writing (Gated)** | `fs.writeFile(path)` | **Protected** | ✅ **YES** (HITL) |
| | `fs.delete(path)` | **Protected** | ✅ **YES** (HITL) |
| | `child_process.exec` | **Protected** | ✅ **YES** (HITL) |
| | `http.post(url)` | **Protected** | ✅ **YES** (HITL) |
| **System (Critical)** | `process.exit()` | **Protected** | ✅ **YES** (HITL) |
| | `npm install` | **Protected** | ✅ **YES** (HITL) |

*   **Autonomous**: Agent can decide to do this freely to gather information.
*   **Protected**: Agent must ask the Guard, who asks the User. Default is DENY.

### 3.3 Approval Workflow
1.  Agent attempts to run `curl https://example.com`.
2.  `RuntimeGuard` intercepts execution.
3.  Approvals Socket (`~/.openclaw/exec-approvals.sock`) emits a **Request**.
4.  Sanctum UI (via Guard) receives the request and prompts the User: *"Agent wants to run `curl`. **Approve / Deny?**"*
5.  User clicks "Approve".
6.  Guard sends `allow-once` back to the socket.
7.  Agent proceeds.

## 4. Container Interface (Docker)

### 4.1 Implementation Constraints
*   **User**: `node` (UID 1000). **Strictly NO Root.**
*   **Hardening**: `cap_drop: [ALL]`, `pids_limit: 100`, `seccomp: agent-profile.json`.
*   **Read-Only Root**: The container root filesystem (`/`) should be mounted read-only (`--read-only`).
*   **Safe Mounts**: `/tmp` and `/dev/shm` must be mounted `noexec,nosuid,nodev`. Writable `workspace/` is accepted risk (mitigated by HITL).

### 4.2 Network Interfaces
*   **`eth0` (Control)**: Connected to `control-net`. Gateway API reachable here.
*   **`eth1` (Execution)**: Connected to `execution-net`. No Gateway; used for CDP connection to Scout.

## 5. Scout Interface (CDP)

Agent Zero connects to the Scout (Browser) using standard **Chrome DevTools Protocol (CDP)**.

*   **Connection**: `ws://scout:9222`
*   **Capability**:
    *   `Target.createTarget(url)`: Open tab.
    *   `Page.captureScreenshot`: Get vision data.
    *   `Runtime.evaluate`: Execute easy-to-verify getters (e.g., `document.body.innerText`).
*   **Restriction**: Agent Zero does **not** proxy traffic through Scout. It only controls the browser via CDP.
