# Analysis: Safe Agent Zero ("Sanctum") Architecture

## Executive Summary
Safe Agent Zero ("Sanctum") is a proposed high-security runtime environment for autonomous agents (specifically OpenClaw/Agent Zero). It employs a **Defense-in-Depth** strategy consisting of 10 layers to mitigate risks associated with autonomous code execution, internet access, and tool usage. The core philosophy is **Zero Trust**, **Default Deny**, and **Private by Default**.

## Key Architectural Components

### 1. The Sanctum Stack (Tiered Isolation)
The architecture uses Docker containerization with strict network segmentation:
-   **Frontend Net**: Public facing (host:443), connects only to the Nginx Guard.
-   **Control Net**: Internal (Guard <-> Agent), used for ACP (Agent Client Protocol) over WebSocket.
-   **Execution Net**: Air-gapped (Agent <-> Scout), used for CDP (Chrome DevTools Protocol). *No internet gateway.*

### 2. The Defense Layers
| Layer | Name | Description |
| :--- | :--- | :--- |
| **0** | Host Access | SSH hardening (Keys only, no root, allowlist). |
| **1** | Host Hardening | Read-only root FS, non-root user (UID 1000), ephemeral `/tmp`. |
| **2** | Network Isolation | No direct internet for Agent. Strict egress whitelisting via Guard. |
| **3** | The Guard (Nginx) | Reverse proxy with Auth (MFA/Basic), Origin validation, and payload limits. |
| **4** | Application Control | `ExecApprovals` via `exec-approvals.json`. "Ask Always" policy for writes/exec. |
| **5** | Data Sanitization | Scout returns accessibility tree/screenshots, not raw HTML (Prompt Injection mitigation). |
| **6** | Audit & Observation | Centralized logging of all inputs/outputs and network traffic. |
| **7** | Anti-Scanning | No public port binding for Agent. Agent only accessible via Guard. |
| **8** | Secret Management | Secrets injected via `.env` (memory only). Never in `config.json` or git. |
| **9** | Integration Locking | Chatbots respond only to whitelisted User IDs. |
| **10** | Agentic Red Teaming | Continuous validation via autonomous "Red Agent" attacks. |

### 3. OpenClaw Interface
-   **Protocol**: Agent Client Protocol (ACP) over Secure WebSocket (WSS).
-   **Port**: 18789.
-   **Security**: Leveraging OpenClaw's native `ExecApprovals` system to enforce HITL (Human-in-the-Loop) for sensitive actions (`fs.writeFile`, `child_process.exec`, etc.).

## Threat Model Mitigation
-   **Indirect Prompt Injection**: Mitigated by Scout's structure-only browsing and screenshot analysis.
-   **Data Exfiltration**: blocked by strict egress filtering and DNS whitelisting.
-   **Container Escape**: Mitigated by Rootless Docker, Read-Only FS, and no Docker socket mounting.

## Status
-   **Docs**: Draft/Planning.
-   **Implementation**: pending full implementation of the `implementation_plan.md`.
