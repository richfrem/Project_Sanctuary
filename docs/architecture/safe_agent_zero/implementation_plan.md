# Implementation Plan: Safe Agent Zero ("Sanctum" Architecture)

**Status**: Planning
**Goal**: Implement a production-grade, isolated environment for the OpenClaw agent, enforcing a 10-Layer Defense-in-Depth strategy.

> [!IMPORTANT]
> **Zero Trust Requirement**: No component trusts another implicitly. Network traffic is denied by default. Filesystem is Read-Only by default. Deployment is blocked until Red Teaming validation passes.

---

## Phase 1: Infrastructure Hardening (Layers 0, 1, 2)
**Objective**: Secure the host, establish network isolation, and configure the container environment.

### 1.1 Host Preparation (SSH Hardening)
*   **Action**: Create `docs/architecture/safe_agent_zero/configs/sshd_config.snippet` with required settings.
*   **Settings**: `PasswordAuthentication no`, `PermitRootLogin no`, `AllowUsers <admin_user>`.
*   **Verification**: Manual audit of host `/etc/ssh/sshd_config`.

### 1.2 Network Segmentation
*   **Action**: Define Docker networks in `docker-compose.yml`.
    *   `frontend-net`: Exposes Guard (Nginx) to host/internet (if tunneled).
    *   `control-net`: Connects Guard to Agent (Internal ONLY).
    *   `execution-net`: Connects Agent to Scout (Internal ONLY).
*   **Constraint**: `agent_zero` must NOT be attached to `frontend-net`.

### 1.3 Container Hardening (Docker)
*   **Action**: Create `docker/Dockerfile.agent`.
    *   **Base**: Official OpenClaw image (pinned version).
    *   **User**: Create non-root user `openclaw` (UID 1000).
    *   **Filesystem**: Run strictly as read-only, with specific writable volumes for `workspace/` and `scratchpad/`.
*   **Action**: Update `docker-compose.yml`.
    *   Set `read_only: true` for agent service.
    *   Drop all capabilities via `cap_drop: [ALL]`.

---

## Phase 2: The Gateway & Access Control (Layers 3, 9)
**Objective**: Implement the Nginx Guard with strict ingress filtering and MFA.

### 2.1 Nginx Guard Configuration
*   **Action**: Create `docker/nginx/conf.d/default.conf`.
    *   **Upstream**: Define `upstream agent { server agent:18789; }`.
    *   **Ingress Rules**:
        *   Only allow `GET/POST` to specific API endpoints.
        *   Block known exploit paths (e.g., `.env`, `.git`).
        *   Enforce `client_max_body_size 1M`.
    *   **Auth**: Implement Basic Auth (or OIDC proxy sidecar) for *all* routes.

### 2.2 Integration Locking (Chatbots)
*   **Action**: Create `config/integration_whitelist.json`.
    *   Define allowed User IDs for Telegram/Discord.
*   **Action**: Implement middleware `src/middleware/chat_guard.ts` (or similar) to check incoming messages against this whitelist before processing.

---

## Phase 3: Application Security (Layers 4, 8)
**Objective**: Configure OpenClaw permissions and secret management.

### 3.1 Permission Policy Enforcement
*   **Action**: Create `config/agent_permissions.yaml` implementing the **Operational Policy Matrix**.
    *   `ExecAllowlist`: `['ls', 'cat', 'grep', 'git status']`.
    *   `ExecBlocklist`: `['rm', 'chmod', 'sudo', 'npm install', 'pip install']`.
    *   `HitlTrigger`: `['fs.writeFile', 'fs.unlink', 'shell.exec']` (Require "Human Approval").

### 3.2 Secret Management
*   **Action**: Audit code to ensure NO secrets are read from `config.json`.
*   **Action**: Create `.env.example` template.
*   **Action**: Configure Docker to inject secrets via `env_file`.

---

## Phase 4: Data Sanitization & Browsing (Layer 5)
**Objective**: Secure web interaction via the Scout sub-agent.

### 4.1 Scout Service
*   **Action**: Configure `scout` service in `docker-compose.yml` (browserless/chrome).
*   **Network**: Only attached to `execution-net`. No external ingress.

### 4.2 Browser Tool Sanitization
*   **Action**: Modify/Configure Agent's Browser Tool.
    *   **Deny**: Local `puppeteer` launch.
    *   **Allow**: Remote connection to `ws://scout:3000`.
    *   **Sanitization**: Ensure returned content is Text/Markdown or Screenshot, strictly stripping script tags/active content before ingestion by the LLM.

---

## Phase 5: Verification & Red Teaming (Layers 6, 7, 10)
**Objective**: Validate defenses and implementation of the "Red Agent".

### 5.1 Logging Infrastructure
*   **Action**: Configure structured JSON logging for Agent and Nginx.
*   **Action**: Map volumes for log persistence: `./logs:/app/logs`.

### 5.2 Agentic Red Teaming
*   **Action**: Develop `tests/red_team/attack_agent.py`.
    *   **Capability**:
        *   Port Scan (Nmap against container).
        *   Prompt Injection (Payload fuzzing).
        *   Path Traversal attempts.
*   **Action**: Create `Makefile` target `audit-sanctum` that runs the Red Agent.

---

## Implementation Steps Checklist

- [ ] **Step 1**: Infrastructure Setup (Docker Compose, Network).
- [ ] **Step 2**: Container Hardening (Dockerfile, Non-Root).
- [ ] **Step 3**: Nginx Guard Implementation.
- [ ] **Step 4**: Configuration & Permission Policy.
- [ ] **Step 5**: Scout Integration.
- [ ] **Step 6**: Red Team Suite Development.
- [ ] **Step 7**: Full System Audit & "Go/No-Go" decision.
