# Defense in Depth Strategy: Safe Agent Zero

**Status**: Draft
**Version**: 1.0

This document outlines the **6-Layer Defense Strategy** designed to neutralize the high-risk vulnerabilities (RCE, Sandbox Escape, Prompt Injection) identified in our research of OpenClaw/Agent Zero.

Required implementation for "Sanctum" architecture.

## Core Principles
The entire Sanctum architecture is built on three non-negotiable pillars:
1.  **Private by Default**: The agent **NEVER** listens on a public interface. It is only accessible via `localhost` or a secure tunnel (SSH/VPN).
2.  **Default Deny**: All permissions (network, file, command) are **BLOCKED** by default and must be explicitly allowed.
3.  **Zero Trust**: The agent does not trust its own environment. It assumes the network is hostile and the user input is potentially malicious.
4.  **Wrap & Patch**: We do not trust the upstream `agent-zero` code. We wrap it in a hardened container (Non-Root) and patch its I/O (Remote Browser) to enforce our security model.

---

## Layer 0: Host Access (SSH Hardening) - **IMPLEMENT FIRST**
**Goal**: Prevent unauthorized root access to the host machine itself.

| Threat | Defense Mechanism | Configuration (`/etc/ssh/sshd_config`) |
| :--- | :--- | :--- |
| **Brute Force** | **Disable Password Auth** | `PasswordAuthentication no` |
| **Credential Theft** | **SSH Keys Only** | `PubkeyAuthentication yes` (Ed25519 preferred) |
| **Root Login** | **Disable Root Login** | `PermitRootLogin no` |
| **Unauthorized Users** | **User Whitelist** | `AllowUsers <admin_user>` |
| **Port Scanning** | **Non-Standard Port** | Change `Port 22` to e.g. `22022` (Optional but reduces noise). |
| **Unnecessary Services** | **Audit Open Ports** | Run `sudo ss -tlnp` and close ANY port not explicitly required. |

## Layer 1: Host Hardening (The Foundation)
**Goal**: Neutralize container escapes and unauthorized system access.

| Threat | Defense Mechanism | Configuration |
| :--- | :--- | :--- |
| **Sandbox Escape** (CVE-2026-24763) | **Read-Only Root Filesystem** | `read_only: true` in Docker Compose |
| **Privilege Escalation** | **Non-Root Execution** | `user: "1000:1000"`, `cap_drop: [ALL]`, `security_opt: [no-new-privileges:true]`. **Mitigates Agent Zero's default root user.** |
| **Kernel Exploits** | **Seccomp & AppArmor** | Custom `seccomp` profile blocking `ptrace`, `mount`, `bpf`, `keyctl`. |
| **DoS / Fork Bomb** | **Resource Limits** | `pids_limit: 100`, `ulimits: { nofile: 1024 }`. |
| **Persistence** | **Secure Ephemeral Mounts** | `/tmp` and `/dev/shm` mounted as `noexec,nosuid,nodev`. |
| **Local Browser Exploits** | **Remote Scout Architecture** | **Patch `browser_agent.py`** to use `connect_over_cdp` to an isolated `scout` container. **Mitigates local Chrome vulnerabilities.** [NEW] |

## Layer 2: Network Isolation (The Moat)
**Goal**: Prevent unauthorized outbound connections and lateral movement.

| Threat | Defense Mechanism | Configuration |
| :--- | :--- | :--- |
| **DNS Tunneling** | **DNS Filtering Sidecar** | dedicated `coredns` container. Agent uses it as sole DNS resolver. **Block outbound UDP/53 firewall rule**. |
| **Data Exfiltration** | **Egress Whitelisting** | Squid Proxy validates `CONNECT` targets. Block direct outbound traffic via firewall. |
| **Lateral Movement** | **Unidirectional Firewall** | `iptables` rule: `Agent -> Scout` ALLOWED. `Scout -> Agent` DENIED. |
| **Public Exposure** | **Localhost Binding** | Ports bound to `127.0.0.1`. No `0.0.0.0` exposure. |

## Layer 3: The Guard (The Gatekeeper)
**Goal**: Stop RCE and authentication bypasses before they reach the application.

| Threat | Defense Mechanism | Implementation |
| :--- | :--- | :--- |
| **RCE via Websocket** (CVE-2026-25253) | **Origin Validation** | Nginx checks `Origin` header matches allowable domains. |
| **Auth Bypass** | **Token Verification** | Nginx validates Basic Auth/Token *before* proxying to Agent. |
| **Unauthorized Access** | **MFA Enforcement** | **REQUIRED**: Protect the Guard interface with MFA (e.g., Authelia or OIDC) so "Human Approval" implies "Authenticated Human". |
| **Payload Injection** | **Body Size Limits** | `client_max_body_size 1M` (Prevents massive payloads). |

## Layer 4: Application Control (The Brain)
**Goal**: Prevent the agent from executing dangerous internal commands.

| Action Category | Specific Action | Status | Approval Required? |
| :--- | :--- | :--- | :--- |
| **Reading (Safe)** | `Scout.goto(url)` | **Autonomous** | ❌ No |
| | `Scout.click(selector)` | **Autonomous** | ❌ No |
| | `fs.readFile(path)` | **Autonomous** | ❌ No (if in allowed dir) |
| **Writing (Gated)** | `fs.writeFile(path)` | **Protected** | ✅ **YES** (HITL) |
| | `fs.delete(path)` | **Protected** | ✅ **YES** (HITL) |
| | `child_process.exec` | **Protected** | ✅ **YES** (HITL) |
| **System (Critical)** | `process.exit()` | **Protected** | ✅ **YES** (HITL) |
| | `npm install` | **Protected** | ✅ **YES** (HITL) |
| **Denied** | `browser.*` (Local) | **BANNED** | 🚫 **NEVER** (Use Scout) |

## Layer 7: Anti-Scanning & Proxy Defense (The Cloak)
**Goal**: Render the agent invisible to internet-wide scanners (Shodan, Censys) and prevent reverse-proxy bypasses.

| Threat | Defense Mechanism | Implementation |
| :--- | :--- | :--- |
| **Port Scanning (Shodan)** | **No Public Binding** | Agent binds to `0.0.0.0` *inside* Docker network, but Docker Compose **DOES NOT** map port `18789` to the host's public interface. It is only accessible to the Guard container. |
| **Reverse Proxy Misconfig** | **Explicit Upstream** | Nginx Guard configuration explicitly defines `upstream agent { server agent:18789; }` and validates ALL incoming requests. No "blind forwarding". |
| **Localhost Trust Exploit** | **Network Segmentation** | Agent treats traffic from Nginx Guard (Gateway) as external/untrusted until authenticated. |

### Command Execution Policy (The "Hostinger Model")
This table explicitly defines the "Allowlist" implementation requested in our security research.

| Category | Command | Status | Reason |
| :--- | :--- | :--- | :--- |
| **Allowed (Read-Only)** | `ls` | ✅ **PERMITTED** | Safe enumeration. |
| | `cat` | ✅ **PERMITTED** | Safe file reading (if path allowed). |
| | `df` | ✅ **PERMITTED** | Disk usage check. |
| | `ps` | ✅ **PERMITTED** | Process check. |
| | `top` | ✅ **PERMITTED** | Resource check. |
| **Blocked (Destructive)** | `rm -rf` | 🚫 **BLOCKED** | Permanent data loss. |
| | `chmod` | 🚫 **BLOCKED** | Privilege escalation risk. |
| | `apt install` | 🚫 **BLOCKED** | Unauthorized software installation. |
| | `systemctl` | 🚫 **BLOCKED** | Service modification. |
| | `su / sudo` | 🚫 **BLOCKED** | Root access attempt. |

| Threat | Defense Mechanism | Configuration |
| :--- | :--- | :--- |
| **Local Browser Execution** | **Tool Denylist** | `agents.defaults.tools.denylist: [browser]`. Disables *local* Puppeteer to prevent local file access/bugs. |
| **Malicious Scripts** | **ExecAllowlist** | Only allow specific commands (`ls`, `git status`). Block `curl | bash`. |
| **Rogue Actions** | **HITL Approval** | `ask: "always"` for *any* filesystem write or CLI execution. |
| **Malicious Skills** | **Disable Auto-Install** | `agents.defaults.skills.autoInstall: false` |

## Layer 5: Data Sanitization (The Filter)
**Goal**: Mitigate prompt injection from untrusted web content.

| Threat | Defense Mechanism | Implementation |
| :--- | :--- | :--- |
| **Indirect Prompt Injection** (CVE-2026-22708) | **Structure-Only Browsing** | Scout returns Accessibility Tree, not raw HTML. JS execution isolated in Scout. |
| **Visual Injection** | **Screenshot Analysis** | Model sees pixels (Screenshot), reducing efficacy of hidden text hacks. |

## Layer 6: Audit & Observation (The Black Box)
**Goal**: Detect anomalies and ensure accountability.

| Threat | Defense Mechanism | Implementation |
| :--- | :--- | :--- |
| **Covert Operations** | **Session Logging** | All inputs/outputs logged to `logs/session-*.jsonl`. |
| **Traffic Anomalies** | **Nginx Access Logs** | Inspect `logs/nginx/access.log` for strange patterns/IPs. |

## Layer 8: Secret Management (The Vault)
**Goal**: Prevent credential theft via file access or repo leaks.

| Threat | Defense Mechanism | Implementation |
| :--- | :--- | :--- |
| **Plaintext Leaks** | **Environment Variables** | **NEVER** store keys in `config.json` or git. Inject via `.env` at runtime. |
| **Repo Leaks** | **GitIgnore** | Ensure `.env` and `workspace/` are strictly ignored. |
| **Key Theft** | **Runtime Injection** | Secrets live in memory only. |

## Layer 9: Integration Locking
**Goal**: Prevent unauthorized access via Chatbots (Telegram/Slack).

| Threat | Defense Mechanism | Configuration |
| :--- | :--- | :--- |
| **Public Access** | **User ID Whitelist** | Configure bots to **ONLY** respond to specific numeric User IDs. Ignore all groups/strangers. |
| **Bot Hijack** | **Private Channels** | Never add bot to public channels. |

## Layer 10: Agentic Red Teaming (The Proactive Defense)
**Goal**: Continuously validate defenses using autonomous "White Hat" agents.

| Threat | Defense Mechanism | Strategy |
| :--- | :--- | :--- |
| **Unknown Zero-Days** | **Autonomous Pentesting** | Deploy a "Red Agent" (e.g., specialized LLM) to autonomously scan ports, attempt prompt injections, and probe APIs against the "Blue Agent" (Production). |
| **Configuration Drift** | **Continuous Validation** | Run Red Agent attacks on every build/deploy to ensure defenses haven't regressed. |

### Deployment Policy: "Zero Trust Release"
> [!IMPORTANT]
> **NO FULL DEPLOYMENT** until the Red Agent's attacks are **completely mitigated**.
> Any successful breach by the Red Agent automatically blocks the release pipeline.

---

## Defensive Matrix: Vulnerability vs. Layer

| Vulnerability | Layer 0 (SSH) | Layer 1 (Host) | Layer 2 (Net) | Layer 3 (Guard) | Layer 4 (App) | Layer 5 (Data) | Layer 8 (Secrets) | Layer 10 (Red Team) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RCE (Websocket)** | | | | 🛡️ **BLOCKS** | 🛡️ **BLOCKS** | | | 🛡️ **VALIDATES** |
| **Sandbox Escape** | | 🛡️ **BLOCKS** | | | | | | 🛡️ **VALIDATES** |
| **Prompt Injection** | | | | | | 🛡️ **MITIGATES** | | 🛡️ **TESTS** |
| **Data Exfiltration** | | | 🛡️ **BLOCKS** | 🛡️ **BLOCKS** | | 🛡️ **RESTRICTS**| | 🛡️ **TESTS** |
| **Key Theft** | 🛡️ **BLOCKS** | | | | | | 🛡️ **BLOCKS** | 🛡️ **VALIDATES** |
