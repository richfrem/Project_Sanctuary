# Operational Policy Matrix: Sanctum / Safe Agent Zero

**Status**: Enforced
**Version**: 1.0

This document serves as the **Single Source of Truth** for all agent permissions. It defines exactly what the agent can do autonomously, what requires human approval, and what is strictly forbidden.

---

## The Policy Table

| Domain | Action Category | Specific Action / Command | Policy Status | Enforcement Mechanism |
| :--- | :--- | :--- | :--- | :--- |
| **Network** | **Egress (Text)** | `Scout.goto(url)` (Read-Only) | 🟢 **AUTONOMOUS** | Scout Sanitization |
| | **Egress (API)** | `curl api.anthropic.com` | 🟢 **AUTONOMOUS** | Nginx Whitelist |
| | **Egress (General)** | `curl google.com` | 🔴 **BLOCKED** | Nginx Firewall |
| | **Ingress** | Incoming Connection to `18789` | 🔴 **BLOCKED** | Docker Internal Net |
| | **P2P / Social** | Connect to `moltbook.com` | 🔴 **BLOCKED** | DNS/Nginx Block |
| **File System** | **Read (Workspace)** | `fs.readFile(./workspace/*)` | 🟢 **AUTONOMOUS** | App Logic |
| | **Read (System)** | `fs.readFile(/etc/*)` | 🔴 **BLOCKED** | Docker Volume Isolation |
| | **Write (Workspace)** | `fs.writeFile(./workspace/*)` | 🟡 **PROTECTED (HITL)** | App `ask: "always"` |
| | **Write (System)** | `fs.writeFile(/etc/*)` | 🔴 **BLOCKED** | Read-Only Root FS |
| | **Delete** | `rm`, `fs.unlink` | 🟡 **PROTECTED (HITL)** | App `ask: "always"` |
| **Command** | **Safe Enumeration** | `ls`, `cat`, `ps`, `top`, `df` | 🟢 **AUTONOMOUS** | ExecAllowlist |
| | **Execution** | `node script.js`, `python script.py` | 🟡 **PROTECTED (HITL)** | App `ask: "always"` |
| | **Package Mgmt** | `npm install`, `pip install` | 🟡 **PROTECTED (HITL)** | App `ask: "always"` |
| | **System Mod** | `chmod`, `chown`, `systemctl` | 🔴 **BLOCKED** | Non-Root User (UID 1000) |
| | **Destruction** | `rm -rf /` | 🔴 **BLOCKED** | Read-Only Root FS |
| **Interactive** | **Browser Tool** | `browser.launch()` (Local) | 🔴 **BLOCKED** | Tool Denylist |
| | **Scout Tool** | `Scout.navigate()` (Remote) | 🟢 **AUTONOMOUS** | Component Architecture |
| **Secrets** | **Storage** | Write to `config.json` | 🔴 **BLOCKED** | Immutable Config |
| | **Access** | Read `process.env.API_KEY` | 🟢 **AUTONOMOUS** | Runtime Injection |

---

## Legend

*   🟢 **AUTONOMOUS**: The agent typically performs this action without user interruption. Security relies on isolation (Docker, Network) and sanitization (Scout).
*   🟡 **PROTECTED (HITL)**: The agent **MUST** pause and request explicit user approval (via MFA-protected UI) before proceeding.
*   🔴 **BLOCKED**: The action is technically impossible due to architectural constraints (Network blocks, Read-only FS, Non-root user).

## Implementation Checklist

- [ ] **Network**: Configure Nginx whitelist for API domains only.
- [ ] **Filesystem**: Mount root as Read-Only in Docker Compose.
- [ ] **User**: Set `user: 1000:1000` in Dockerfile.
- [ ] **App Config**: Set `agents.defaults.permissions.ask: "always"`.
- [ ] **Tools**: Add `browser` to `agents.defaults.tools.denylist`.
- [ ] **Monitoring**: Ensure `logs/session.jsonl` captures all Yellow/Red attempts.
