

# Round 5: Architecture Review — Safe Agent Zero ("Sanctum")

**Date:** 2026-02-15
**Reviewer:** Claude Opus 4.6 (Senior Software Architect Persona)
**Scope:** Complexity, Scalability, Design Patterns, Maintainability
**Classification:** Protocol 128, Phase IV, Round 5

---

## Architectural Impact Assessment: **High**

The Sanctum architecture has evolved through 4 rounds of security hardening into a multi-container, multi-network, multi-proxy system with sidecars, firewalls, and interception layers. This review evaluates whether the accumulated security controls have created an architecture that is **operationally sustainable** or whether it has crossed into **accidental complexity**.

---

## Pattern Compliance Checklist

- [x] Adherence to existing patterns (Docker Compose, reverse proxy, network segmentation)
- [x] SOLID Principles (each container has a single responsibility)
- [ ] Dependency Management (implicit coupling between 6+ containers creates fragile startup ordering)
- [ ] Separation of Concerns (Scout now serves dual duty: browsing tool AND exfiltration vector requiring its own proxy)

---

## Architectural Critiques

### ARCH-01: Sidecar Proliferation — Approaching Distributed Monolith Territory

**Severity:** High (Architectural)

The current container topology after Round 4 remediations:

| Container | Network(s) | Purpose |
|:----------|:-----------|:--------|
| `nginx-guard` | `frontend-net`, `control-net` | Ingress gateway, auth, rate limiting |
| `agent` | `control-net`, `execution-net` | OpenClaw agent runtime |
| `scout` | `execution-net`, (internet) | Headless browser |
| `coredns` | `control-net` | DNS filtering sidecar |
| `squid` | `execution-net`, (internet) | HTTP/S egress proxy for Agent |
| `scout-proxy` (proposed) | `execution-net`, (internet) | Egress monitor for Scout |

Plus iptables rules, seccomp profiles, and potentially AppArmor.

**The Problem:** For a **single-host, single-user** development tool, we now have 5-6 containers, 3 Docker networks, firewall rules that depend on Docker-assigned IPs, and a DNS sidecar. Each container adds:
- A startup dependency
- A failure mode
- A log stream to monitor
- A configuration file to maintain

This is the **Kubernetes sidecar pattern** applied to a Docker Compose stack that runs on a MacBook. The operational overhead is disproportionate to the deployment context.

**The Trade-off Question:** Is this complexity justified? For running an experimental agent with real API keys — **yes, mostly.** The security posture genuinely requires network segmentation, DNS filtering, and egress control. But the *implementation pattern* can be simplified.

**Recommendation:** Consider collapsing the proxy stack. Instead of separate CoreDNS + Squid + Scout-Proxy containers, use a **single egress gateway container** that handles:
1. DNS resolution (with filtering)
2. HTTP/S proxying (with domain allowlisting)
3. Traffic logging

This reduces container count from 6 to 4 (Guard, Agent, Scout, Egress-Gateway) while preserving the same security properties. Tools like `mitmproxy` or `nginx` with stream modules can serve all three functions.

---

### ARCH-02: The CoreDNS Sidecar — Over-Engineered for Single-Host

**Severity:** Medium-High (Pattern Validity)

CoreDNS is a production-grade Kubernetes DNS server designed for cluster-scale service discovery. Using it as a filtering DNS resolver for a single container is architecturally valid but operationally heavy:

- CoreDNS requires a `Corefile` configuration
- It exposes health/metrics endpoints that themselves become attack surface (Round 4 finding SIDECAR-02)
- It introduces a new failure mode (crash → potential DNS fallback, finding SIDECAR-01)
- It requires explicit container restart policies and anti-fallback DNS options on the Agent

**Alternative Pattern: Host-Level DNS Masquerading**

For a single-host deployment, the simpler pattern is:

1. **No DNS sidecar at all.** Instead, override the Agent's `/etc/resolv.conf` to point at a non-routable IP.
2. Use **iptables DNAT rules** on the host to redirect DNS from the Agent to a local `dnsmasq` process (or even just `/etc/hosts` injection for the 3-4 whitelisted domains).
3. Block all other DNS traffic at the firewall level.

This eliminates an entire container, its configuration, its failure modes, and its attack surface. The trade-off is that it couples the DNS filtering to the host rather than making it portable — but this is a single-user development tool, not a distributed system.

**If CoreDNS is retained**, the architecture should at minimum:
- Bind health/metrics to `127.0.0.1` inside the CoreDNS container (as Round 4 recommended)
- Set `restart: always` 
- Use a minimal Corefile with only the `forward` and `hosts` plugins (no `prometheus`, no `cache` if not needed)

**Verdict:** CoreDNS is defensible but not optimal. For a v1 prototype, `dnsmasq` or host-level iptables DNAT is simpler, fewer moving parts, and equally secure.

---

### ARCH-03: The Navigation Guard — CDP Interception Is Architecturally Fragile

**Severity:** High (Pattern Validity)

Round 4 proposed intercepting Chrome DevTools Protocol (CDP) commands between Agent and Scout to filter `Page.navigate` calls to non-whitelisted domains. The Implementation Plan (Phase 4.2) now includes "Navigation Guard: Intercept `Page.navigate` CDP commands."

**Why This Is a Hack, Not a Pattern:**

1. **CDP is not designed for interception.** It's a debugging protocol with hundreds of commands. A WebSocket proxy that parses CDP messages is fragile — command formats change between Chrome versions, and there are multiple ways to navigate (not just `Page.navigate`):
   - `Target.createTarget({ url })` 
   - `Runtime.evaluate("window.location = 'url'")`
   - `Runtime.evaluate("fetch('url')")`
   - `Page.setDownloadBehavior` + link click
   - `Input.dispatchMouseEvent` on a link element
   
2. **Maintenance burden is high.** Every Chrome/Chromium update could break the interception proxy. The CDP protocol is versioned and evolving.

3. **False sense of security.** Even if you intercept `Page.navigate`, `Runtime.evaluate` can execute arbitrary JavaScript in the browser context, including `fetch()` calls, form submissions, and WebSocket connections. Filtering CDP comprehensively is equivalent to building a JavaScript sandbox inside the browser — which is the browser's job, not ours.

**Better Pattern: Remote Browser Isolation (RBI)**

The industry-standard pattern for "agent controls a browser but browser can't exfiltrate" is **Remote Browser Isolation**:

1. Scout runs in a fully isolated network with **NO direct internet access**.
2. A dedicated **browsing proxy** (e.g., Squid, or a custom relay) mediates ALL Scout traffic.
3. The proxy logs every URL, blocks non-whitelisted domains, and can inspect response content.
4. The Agent connects to Scout via CDP as before — no interception needed at the CDP layer.

This is architecturally cleaner because:
- **The network enforces isolation**, not application-layer command parsing
- Scout's traffic goes through a proxy regardless of how the navigation was triggered
- No fragile CDP parsing required
- Standard pattern used by enterprise browser isolation products (Cloudflare, Zscaler, etc.)

**The Implementation Plan should replace "Navigation Guard" (Phase 4.2) with:**
1. Remove Scout's direct internet access (detach from any external network)
2. Add Scout to a `browsing-net` with a dedicated forward proxy
3. Proxy logs all URLs and enforces domain allowlist
4. Agent → CDP → Scout → Proxy → Internet

This is essentially Round 4's "Option D" (Scout Network Isolation + Relay), which was noted as "most secure" but dismissed for "adding complexity." I'd argue it's actually **less complex** than CDP interception because it uses standard networking patterns instead of protocol-specific parsing.

---

### ARCH-04: Startup Ordering and Health Check Cascade

**Severity:** Medium (Operational)

The `operational_workflows.md` specifies a boot sequence: Networks → Guard → Agent → Scout. Docker Compose `depends_on` handles basic ordering but **not health verification**.

With 5-6 containers and cross-network dependencies:
1. CoreDNS must be healthy before Agent starts (Agent's DNS won't resolve otherwise)
2. Squid must be healthy before Agent makes API calls
3. Guard must be healthy before any external access
4. Scout must be healthy before Agent attempts CDP connection

Docker Compose `depends_on` with `condition: service_healthy` requires health checks on every service. The current plan doesn't specify health check definitions for CoreDNS, Squid, or Scout.

**Failure Mode:** If CoreDNS starts but its Corefile has a syntax error, it may report "healthy" (port 53 open) but not resolve any queries. The Agent starts, can't reach APIs, and enters an error loop.

**Recommendation:**
- Define explicit `healthcheck` blocks for every service in `docker-compose.yml`
- CoreDNS: `dig @127.0.0.1 api.anthropic.com` returns expected IP
- Squid: `squidclient -h 127.0.0.1 mgr:info` returns valid response
- Scout: `curl http://localhost:9222/json/version` returns Chrome version
- Add a top-level `startup.sh` script that verifies the full chain (Agent → CoreDNS → Squid → Internet) before declaring ready

---

### ARCH-05: Configuration Fragmentation

**Severity:** Medium (Maintainability)

Security configuration is currently spread across:

| Config | Location | Format |
|:-------|:---------|:-------|
| Docker Compose | `docker-compose.yml` | YAML |
| Seccomp profile | `docker/seccomp/agent-profile.json` | JSON |
| Nginx config | `docker/nginx/conf.d/default.conf` | Nginx conf |
| Squid config | `squid.conf` (location unspecified) | Squid conf |
| CoreDNS config | `Corefile` (location unspecified) | CoreDNS DSL |
| ExecApprovals | `config/exec-approvals.json` | JSON |
| Integration whitelist | `config/integration_whitelist.json` | JSON |
| Agent permissions | `config/agent_permissions.yaml` | YAML |
| iptables rules | Unspecified (startup script?) | Shell |

That's **9 configuration files in 5 different formats** across at least 3 directories. A single misconfiguration in any one of these can compromise the security posture.

**Recommendation:** 
- Consolidate all configuration files under `docker/config/` with a clear naming convention
- Create a `make verify-config` target that validates all configs before `docker compose up`
- Consider generating derived configs (iptables rules, DNS forwarding lists) from a single source of truth (e.g., a `sanctum-policy.yaml` that lists allowed domains, and scripts generate CoreDNS zones + Squid ACLs + iptables rules from it)

---

## Scalability Assessment

### For the Stated Use Case (Single User, Single Host): Adequate

The architecture doesn't need horizontal scaling. A MacBook running 5-6 containers with resource limits (512MB RAM, 1 CPU for Agent) is well within hardware capability. The main scalability concern is **cognitive scalability** — can a developer (or future maintainer) understand and debug this system?

### Cognitive Scalability: At Risk

A new developer approaching this system must understand:
- Docker networking (3 networks, inter-container routing)
- iptables (unidirectional rules, DNS blocking)
- Squid proxy configuration (CONNECT ACLs, explicit vs transparent mode)
- CoreDNS (Corefile syntax, plugin system)
- Seccomp profiles (syscall filtering)
- CDP protocol (how Agent controls Scout)
- OpenClaw's permission system (ExecApprovals)
- Nginx reverse proxy (auth, rate limiting, upstream routing)

That's **8 distinct technology domains** for what is fundamentally "run an agent in a box." Each domain has its own debugging tools, log formats, and failure modes.

**Recommendation:** Create a `docs/architecture/safe_agent_zero/OPERATIONS_RUNBOOK.md` that covers:
1. How to verify each layer is working (commands + expected output)
2. Common failure modes and their symptoms
3. How to add a new domain to the allowlist (single checklist touching all relevant configs)
4. How to read each log format

---

## Alternative Patterns Considered

### Pattern A: Firecracker microVM (Grok's Recommendation)

**Trade-off:** Strongest isolation but requires KVM support (not available on macOS without nested virtualization). Not viable for the primary deployment target (MacBook).

**Verdict:** Correct for Linux production servers. Not applicable here.

### Pattern B: gVisor (runsc)

**Trade-off:** User-space kernel that intercepts syscalls. Excellent for untrusted workloads. Eliminates need for seccomp (gVisor handles syscall filtering). Available on macOS via Docker Desktop.

**Verdict:** Would simplify the architecture by replacing seccomp + most iptables rules with a single runtime configuration. Worth investigating for v2.

### Pattern C: Single Egress Gateway (Recommended Simplification)

Replace CoreDNS + Squid + Scout-Proxy with a single `mitmproxy` or `nginx stream` container that handles DNS filtering, HTTP/S proxying, and traffic logging.

**Trade-off:** Slightly less defense-in-depth (single point of failure for egress) but dramatically simpler to operate and debug.

**Verdict:** Best trade-off for v1 prototype. Can be decomposed into separate services later if needed.

---

## Verdict: Is the Design Sound?

**Yes, with reservations.**

The **security design** is thorough, well-researched, and addresses real threats. The 4 rounds of red teaming have produced a genuinely hardened architecture. The team should be commended for the rigor.

The **implementation design** has accumulated complexity through successive hardening rounds without consolidation. Each round added controls (CoreDNS sidecar, Squid proxy, Navigation Guard, Scout proxy, iptables rules) without questioning whether previous additions could be simplified or merged.

### Specific Recommendations (Prioritized)

| Priority | Action | Impact |
|:---------|:-------|:-------|
| **P0** | Replace "Navigation Guard" (CDP interception) with Scout network isolation + browsing proxy | Eliminates fragile protocol-specific filtering; uses standard networking patterns |
| **P1** | Consolidate CoreDNS + Squid into single egress gateway container | Reduces container count, simplifies config, fewer failure modes |
| **P1** | Create unified policy file that generates derived configs | Single source of truth for allowed domains across DNS/proxy/firewall |
| **P2** | Add health checks and startup verification script | Prevents silent misconfiguration failures |
| **P2** | Write operations runbook | Enables maintainability beyond the original architect |
| **P3** | Evaluate gVisor as runtime for v2 | Could replace seccomp + simplify host hardening |

### The Core Tension

This architecture embodies a tension between **security depth** and **operational simplicity.** Both matter. An architecture that is secure but impossible to operate correctly will be misconfigured in practice — which is worse than a simpler architecture that's easy to get right.

The current design is at the edge of that trade-off. One more round of "add a sidecar for X" would tip it over. **The next iteration should consolidate, not add.**

---

*End of Round 5 Architecture Review*
*Reviewer: Claude Opus 4.6 (Senior Software Architect Persona)*
*Protocol 128, Phase IV, Round 5*

---

Shall I write this to `docs/architecture/safe_agent_zero/red_team_reviews/REAL_RED_TEAM_ROUND_5_FINDINGS.md`?
