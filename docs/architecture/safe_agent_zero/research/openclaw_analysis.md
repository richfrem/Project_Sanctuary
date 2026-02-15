# OpenClaw Architecture & Security Analysis

**Source**: `docs/architecture/safe_agent_zero/research/openclaw`
**Date**: 2026-02-15

## 1. Executive Summary

OpenClaw is a Node.js/TypeScript-based autonomous agent framework designed with a "Gateway" architecture. It explicitly supports a **Sandbox Mode** and has robust, built-in security mechanisms for **Command Execution (HITL)** and **Network Isolation**.

Its architecture aligns perfectly with the "Sanctum" strategy, specifically its ability to run as a non-root container and its built-in approval hooks.

## 2. Security Architecture

### A. Execution Guardrails (`src/infra/exec-approvals.ts`)
OpenClaw implements a strict **Execution Approval** system:
*   **Modes**: `deny`, `allowlist`, `full`.
*   **Default**: `deny` (safe by default).
*   **Mechanism**: Every shell command is analyzed. If it's not in the allowlist, it triggers an approval request (HITL).
*   **Persistence**: Approvals are stored in `~/.openclaw/exec-approvals.json` (SHA-256 hashed).
*   **Socket Control**: Approvals can be requested via a Unix socket (`~/.openclaw/exec-approvals.sock`), allowing external tools (like our Nginx Guard?) to potentially interact with it.

### B. Containerization (`Dockerfile`)
*   **User**: Runs as `node` (UID 1000), **not root**. This mitigates container escape risks.
*   **Base Image**: `node:22-bookworm` (Debian 12).
*   **Sandbox**: There is a dedicated `Dockerfile.sandbox` using `debian:bookworm-slim` with minimal tools (`curl`, `git`, `jq`, `python3`, `ripgrep`).

### C. Network & Gateway (`src/infra/gateway-lock.ts`)
*   **Loopback Binding**: By default, the Gateway binds to `127.0.0.1`.
*   **Traffic**: It uses a standard HTTP/WebSocket interface for clients.
*   **Warning**: `SECURITY.md` explicitly warns *against* exposing the Gateway to the public internet without a reverse proxy (confirming our Nginx Guard decision).

## 3. Integration Points for Sanctuary

1.  **Command Interception**: The `ExecApprovals` module suggests we can configure OpenClaw to **require approval** for *any* command execution. We can map this to a UI feature in the Sanctuary interface.
2.  **ACP Bridge**: `docs.acp.md` describes the "Agent Client Protocol" bridge. This is the ideal protocol for our "Control Network" to communicate with Agent Zero.
3.  **Logs**: Logs are written to `~/.openclaw/sessions/`. Integrating these into the Sanctuary dashboard is straightforward via volume mounts.

## 4. Risks & Mitigations

*   **Node.js Runtime**: Requires Node 22+.
*   **Tool Power**: The `exec` tool is powerful. While `ExecApprovals` restricts *what* runs, a whitelisted `curl` can still hit internal endpoints if not network-gated.
    *   *Mitigation*: Our **Network Segmentation** strategy (`execution-net`) remains critical. The application-layer checks (`exec-approvals`) are a second line of defense, not a replacement for network isolation.

## 5. Conclusion

OpenClaw is "Safe Agent Ready". We do not need to fork it or rewrite its core loop. We simply need to:
1.  **Configure**: Set `security=allowlist` or `deny` in `exec-approvals.json`.
2.  **Containerize**: Use the official `Dockerfile` patterns but wrap them in our isolated Docker Compose stack.
3.  **Bridge**: Use the ACP protocol for control.
