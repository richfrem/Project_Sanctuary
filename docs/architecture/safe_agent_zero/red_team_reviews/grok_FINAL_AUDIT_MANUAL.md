# Safe Agent Zero (Sanctum) Red Team Review — Deep Dive (Iteration 4+)

**Date:** 2026-02-15  
**Reviewer:** Grok 4 (Adversarial Security Researcher — Assume Breach Mindset)  
**Scope:** Full architecture as presented in the audit packet, with emphasis on post-Round-5 MVSA (4-container model), consolidated remediations, seccomp profile, network topology, and operational realities. This review goes beyond prior Claude Opus rounds by stress-testing assumptions, simulating multi-step attacks, and evaluating long-term operational viability.  
**Classification:** Protocol 128 Phase IV — Deep Architectural Validation

---

## Executive Summary

The Sanctum team has done **impressive** work. The evolution from early naïve designs to the current MVSA (Minimum Viable Secure Architecture) shows genuine learning: container escape is now extremely difficult, direct exfiltration is well-controlled, and resource abuse is contained. The seccomp profile is solid (deny-by-default with thoughtful allows), network segmentation is principled, and the consolidation to 4 containers reduces the attack surface dramatically.

**However, "safe" is relative.**

Assuming an intelligent, persistent adversary (prompt-injected agent, supply-chain compromise, or zero-day in Node.js/Chromium), there are **three high-impact pathways** that remain viable:

1. **Scout as Exfiltration Oracle** — Even with isolated browsing-net + proxy, the Agent retains too much control over Scout's behavior.
2. **Sidecar Trust Collapse** — The unified security-sidecar is now a single point of failure with elevated trust.
3. **Operational Drift & Complexity Debt** — The architecture is verifiably secure in theory, but the implementation surface (docker-compose, startup scripts, config files) introduces drift risk.

Previous rounds stopped at "safe to build." This review asks: **Is it safe to run autonomously with real secrets?**

**Verdict: CONDITIONAL SAFE TO BUILD, NOT YET SAFE FOR UNATTENDED AUTONOMOUS OPERATION**

Proceed to implementation, but mandate **three additional controls** before any run with production credentials.

---

## 1. Scout Exfiltration — The Persistent Oracle Problem

### Prior Remediation Recap
- Round 4 identified direct CDP navigation exfiltration.
- Round 5 moved Scout to isolated `browsing-net` with dedicated proxy.
- MVSA keeps Scout internet-detached except through sidecar proxy.

### Remaining Attack Surface

Even with network isolation, the Agent still has **full CDP control** over Scout. This is intentional (the Agent needs to browse), but it creates an **oracle channel**.

#### Attack Scenario 1: Steganographic Exfiltration via Screenshots
1. Agent is compromised (prompt injection via a malicious page Scout visited earlier).
2. Agent reads secrets (`process.env`, workspace files).
3. Agent instructs Scout (via CDP) to:
   - `Runtime.evaluate` JavaScript to render secret data as QR code or high-contrast text on a blank page.
   - `Page.captureScreenshot` → returns base64 PNG to Agent.
4. Agent now has the screenshot in memory.
5. Agent cannot directly exfiltrate the PNG… **but** it can repeat the process: render the PNG as an `<img>` on another page and navigate to a tracker domain with a unique subdomain per pixel chunk (DNS exfiltration), or use timing channels.

**Exploitability:** Medium-High  
**Impact:** Full secret disclosure  
**Why previous fixes don't close it:** Proxy only filters HTTP/S traffic. Screenshots are returned over CDP (internal WebSocket, no proxy). The exfiltration can be slow (DNS/timing) but reliable.

#### Attack Scenario 2: Scout as Covert Channel via Legitimate Navigation
1. Agent navigates Scout to a whitelisted domain it controls (e.g., a GitHub Pages site under attacker control if `*.github.io` is allowed).
2. Uses `Runtime.evaluate` to inject JavaScript that beacons data via:
   - Image loads with query params
   - WebSocket to attacker's server (if WS is not blocked in proxy)
   - Fetch/XHR with encoded data in headers

**Exploitability:** High if any wildcard domains are whitelisted  
**Mitigation Gap:** Proxy ACLs are typically domain-based, not content-based.

### Recommended Hardening (Mandatory for Autonomous Runs)
1. **CDP Command Allowlist Proxy**: Insert a lightweight WebSocket proxy between Agent and Scout that inspects and blocks dangerous CDP methods (`Page.navigate` to non-whitelisted, `Runtime.evaluate` beyond a safe subset, `Page.captureScreenshot` if not needed).
2. **Domain Pinning**: No wildcards. Explicit FQDN whitelist only.
3. **Screenshot Sanitization**: If screenshots are returned to Agent, scan for high-entropy patterns or QR codes server-side (in Guard) before allowing further use.
4. **Alternative**: Replace direct CDP with a mediated browsing API (Agent requests URL → Guard approves → Scout navigates → returns sanitized DOM/screenshot).

---

## 2. Unified Security Sidecar — Single Point of Trust Collapse

### Prior Remediation Recap
- Consolidated CoreDNS + Squid into one sidecar to reduce sprawl.

### New Risk Introduced

The sidecar now has:
- Access to all internal networks
- Internet egress
- DNS resolution authority
- Proxy authority for both Agent and Scout

If the sidecar is compromised (zero-day in Squid, misconfiguration, supply-chain), **the entire isolation model collapses**.

#### Attack Scenario: Sidecar Pivot
1. Attacker exploits a vulnerability in Squid (e.g., CVE in CONNECT handling).
2. Gains code execution inside sidecar (likely root, as proxies often run privileged).
3. From sidecar:
   - Sniff all Agent → Scout traffic
   - MITM Scout's browsing
   - Directly exfiltrate (has internet)
   - Pivot to Agent container (same networks, can reach Agent ports if any exposed)

**Exploitability:** Low-Medium (depends on Squid version maintenance)  
**Impact:** Total compromise

### Recommended Hardening
1. **Least Privilege Sidecar**: Run Squid/Dnsmasq as non-root. Use separate users.
2. **Separate Egress Proxies**: Keep Agent proxy and Scout proxy in different containers (accept slightly more complexity for defense-in-depth).
3. **Sidecar Seccomp/AppArmor**: Apply same rigorous profile as Agent.
4. **Automated Vulnerability Scanning**: CI scan for sidecar images.

---

## 3. Operational Drift & Human Factors

The architecture is theoretically sound, but the **implementation surface** is large:

- docker-compose.yml (not shown in packet, but referenced)
- Startup scripts for iptables rules
- Config files for Squid ACLs, Dnsmasq hosts
- Seccomp JSON maintenance

### Risks
1. **Configuration Drift**: Manual edits to ACLs over time → accidental wildcard.
2. **Verification Gap**: No automated test suite for "can Agent reach 8.8.8.8 directly?" or "can Scout navigate to attacker.com?"
3. **Restart Resilience**: If sidecar restarts, does DNS resolution fail-closed?

### Recommended Controls
1. **Infrastructure as Code Validation**: Git-enforced config with CI tests that spin up the stack and run adversarial checks (curl from Agent to blocked sites, nmap scans, etc.).
2. **Fail-Closed Defaults**: Configure sidecar restart: always, and Agent/Scout to timeout gracefully on proxy/DNS failure.
3. **Immutable Images**: Build sidecar as immutable image, no runtime config changes.

---

## 4. Positive Assessment — What Is Actually Hardened Well

| Area                        | Strength                                                                 |
|-----------------------------|--------------------------------------------------------------------------|
| Container Escape            | Seccomp + cap_drop ALL + no-new-privileges (when added) + read-only root = extremely strong |
| Direct Network Exfiltration | Network segmentation + proxy enforcement = solid                          |
| Resource DoS                | pids_limit, mem_limit, ulimits = contained                                |
| Persistence                 | noexec mounts + limited writable paths = good (interpreted code risk accepted as inherent) |
| Overall Design Evolution    | MVSA consolidation shows maturity                                         |

---

## Final Verdict & Go/No-Go

**Safe to Build:** YES. The current MVSA design is the right target.

**Safe for Unattended Autonomous Operation with Real Secrets:** NO — not yet.

**Required Before Autonomous Runs:**
1. CDP command filtering or mediated browsing API
2. Strict FQDN-only whitelisting (no wildcards)
3. Sidecar least-privilege + separate proxy instances if possible
4. Automated security validation test suite

With these, Sanctum can be considered **production-grade isolated agent architecture**.

Without them, it remains **development-grade** — suitable for prototyping and HITL workflows, but not for long-running autonomous tasks with access to real API keys.

The project is on the correct trajectory. This is one of the most thoughtfully hardened local agent designs I've reviewed.

Your move — shall we draft the CDP proxy spec, write the adversarial CI tests, or simulate a full assume-breach exercise?
