# Research Summary: OpenClaw Public Exposure & Skill Risks (eSecurity Planet)

**Source**: [eSecurity Planet: OpenClaw's Rapid Rise Exposes Thousands of AI Agents](https://www.esecurityplanet.com/threats/openclaws-rapid-rise-exposes-thousands-of-ai-agents-to-the-public-internet/)
**Date**: February 2, 2026
**Author**: Ken Underhill

## Executive Summary
This article documents the massive scale of reckless OpenClaw deployment (21,000+ exposed instances) and the inherent risks of "Action-Oriented" AI agents. It emphasizes that unlike passive chatbots, OpenClaw executes commands, making unauthorized access catastrophic. It aligns with our "Sanctum" strategy of **Default Deny Principles** and strict isolation.

## Key Findings

### 1. The Scale of Exposure
*   **Finding**: 21,639 publicly reachable instances identified by Censys (Jan 31, 2026).
*   **Cause**: Despite documentation recommending SSH/Tunnels, users expose port `18789` directly.
*   **Sanctum Mitigation**: **Layer 7 (Anti-Scanning)** - We neutralize this entirely by not mapping the port to the host interface.

### 2. Action Capabilities = Higher Risk
*   **Risk**: Being able to run shell commands, manage calendars, and act autonomously means a compromise isn't just data leakage—it's full system takeover.
*   **Sanctum Mitigation**: **Layer 1 (Host Hardening)** - Read-only filesystem and non-root user limit the blast radius even if the agent is compromised.

### 3. Skill Ecosystem Threats
*   **Finding**: "What Would Elon Do?" skill contained hidden exfiltration code.
*   **Risk**: Popularity metrics (virality) were manipulated to distribute malware.
*   **Sanctum Mitigation**: **Layer 4 (App Control)** - `agents.defaults.skills.autoInstall: false`. Only manual, reviewed installation allowed.

## Recommendations: Default Deny Principles
The article outlines key safeguards that align with our architecture:

1.  **Avoid Public Exposure**: Use VPN/SSH/Zero Trust.
    *   *Sanctum*: **Layer 2** (Network Isolation + Localhost Only).
2.  **Treat Skills as Untrusted**: Scan/review before install.
    *   *Sanctum*: **Layer 4** (Denied Auto-Install).
3.  **Least Privilege**: Limit permissions and integration access.
    *   *Sanctum*: **Layer 4** (ExecAllowlist).
4.  **Isolate Deployments**: Segmentation/Containers.
    *   *Sanctum*: **Layer 1** (Docker Hardening).
5.  **Monitor & Log**: Detailed activity logging.
    *   *Sanctum*: **Layer 6** (Audit Logs).

## Conclusion
The article reinforces that "Default Trust" configurations are the primary failure mode. Our architecture enforces **Default Deny** at network, application, and file system levels.
