# Research Summary: 40,000+ Vulnerable OpenClaw Agents (TechZine)

**Source**: [TechZine: Over 40,000 OpenClaw agents vulnerable](https://www.techzine.eu/news/security/138633/over-40000-openclaw-agents-vulnerable/)
**Date**: Feb 9, 2026
**Author**: Berry Zwets

## Executive Summary
SecurityScorecard researchers identified over **40,000 unsecured OpenClaw instances** exposed to the public internet across 76 countries. 12,812 are vulnerable to Remote Code Execution (RCE). This establishes OpenClaw as a high-risk asset requiring immediate remediation and strict isolation.

## Key Findings

### 1. Massive Attack Surface
*   **Scale**: 40,214 confirmed instances via favicon fingerprinting.
*   **Vulnerability**: 12,812 instances vulnerable to RCE (CVE-2026-25253).
*   **Impact**: Full system takeover (root file access, SSH keys, credentials).

### 2. Critical CVEs (The "Big Three")
*   **CVE-2026-25253 (CVSS 8.8)**: 1-click RCE via malicious link (Gateway token theft).
*   **CVE-2026-25157 (CVSS 7.8)**: SSH command injection in macOS app.
*   **CVE-2026-24763 (CVSS 8.8)**: Docker sandbox escape via PATH manipulation.
*   **Sanctum Mitigation**: Our architecture isolates the agent in a hardened Docker container (non-root, read-only FS) behind an Nginx Guard, neutralizing all three vectors.

### 3. Default Configuration Failure
*   **Root Cause**: Default binding to `0.0.0.0:18789` exposes the control panel to the world.
*   **Sanctum Mitigation**: We bind strictly to `127.0.0.1` inside a Docker network.

### 4. Sector-Wide Risk
*   **Targets**: Financial services, healthcare, government, and tech sectors are all running exposed agents.
*   **Data at Risk**: API keys, OAuth tokens, browser profiles, crypto wallets.

## Conclusion
The article confirms that "default" OpenClaw deployment is negligent. The only safe way to run OpenClaw is within a **Zero Trust** architecture like Sanctum, which assumes the agent itself is vulnerable and wraps it in external defenses.
