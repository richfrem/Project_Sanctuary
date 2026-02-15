# Research Summary: OpenClaw Security Checklist (Hostinger)

**Source**: [Hostinger Tutorials: OpenClaw Security Checklist](https://www.hostinger.com/ca/tutorials/openclaw-security)
**Date**: Feb 2026
**Author**: Larassatti D.

## Executive Summary
This article provides a practical **13-point security checklist** for self-hosting OpenClaw. It shifts focus from specific CVEs to operational best practices (OpsSec). It emphasizes that "mistakes don't stay confined to a chat window" with agentic AI.

## The 13-Point Checklist vs. Sanctum Strategy

| Checklist Item | Sanctum Implementation | Status |
| :--- | :--- | :--- |
| **1. Keep Private (Localhost)** | **Layer 2**: Bind to `127.0.0.1`. No public IPs. | ✅ Covered |
| **2. Close & Audit Ports** | **Layer 7**: Docker isolation. No host port mapping for `18789`. | ✅ Covered |
| **3. Harden SSH** | **Host Level**: Standard server requirement (outside agent scope). | ⚠️ Ops Req |
| **4. Never Run as Root** | **Layer 1**: User `1000:1000` enforced in Dockerfile. | ✅ Covered |
| **5. Restrict via Allowlist** | **Layer 4**: `ExecAllowlist` for shell commands. | ✅ Covered |
| **6. Require HITL Approval** | **Layer 4**: `ask: "always"` for critical actions. | ✅ Covered |
| **7. Safe Secret Storage** | **Env Vars**: Secrets injected via `.env`, never in config files. | ✅ Covered |
| **8. Docker Isolation** | **Layer 1**: Full containerization + Read-Only FS. | ✅ Covered |
| **9. Browser/Msg Caution** | **Layer 5**: Scout sanitization + "Default Deny" browser tool. | ✅ Covered |
| **10. Lock Down Chat Bots** | **Configuration**: Whitelist User IDs only (no public channels). | ✅ Config Req |
| **11. Audit Logging** | **Layer 6**: JSON logging enabled. | ✅ Covered |
| **12. Update Safely** | **Ops**: Pin Docker image versions. | ✅ Covered |
| **13. Slow Rollout** | **Process**: Use "Read-Only" mode initially. | ✅ Process |

## Key Insights
*   **"Cognitive Context Theft"**: Attackers don't just steal keys; they steal the entire workflow context (chat history, behavioral patterns).
*   **VPS Hardening**: A weak host compromises the strongest container. The underlying VPS must be secured first (SSH keys, Firewall).

## Conclusion
The Hostinger checklist serves as an excellent **Operational Readiness Guide**. While our Architecture ("Sanctum") covers the technical controls (Docker, Network, Guard), the operational controls (SSH hardening, User ID whitelisting) must be part of the deployment procedure.
