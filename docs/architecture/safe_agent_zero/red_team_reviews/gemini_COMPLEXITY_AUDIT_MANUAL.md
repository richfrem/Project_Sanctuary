# 🛡️ FULL_RED_TEAM_COMPLEXITY_AUDIT.md

**Date:** 2026-02-15
**Project:** Sanctuary — Safe Agent Zero (Sanctum)
**Audit Scope:** MVSA (Minimum Viable Secure Architecture) - 4 Container Model

---

## 👥 The Panel

* **The Pragmatist (Distinguished Engineer):** Focuses on shipping working software. Hates "architecture astronauts."
* **The Paranoid (CISO):** Focuses on Zero Trust. Assumes the agent is already compromised.
* **The Operator (SRE):** Focuses on observability, maintenance, and the 3 AM wake-up call.

---

## 1. The Complexity Trap

**The Pragmatist:** We started with a 6-container distributed monolith in the earlier phases. That was a trap. Moving to the 4-container MVSA (Guard, Agent, Scout, Sidecar) is the right call. The "Unified Sidecar" handling both CoreDNS and Squid proxying keeps the `docker-compose.yml` portable without adding unnecessary network hops. We don't need a service mesh for a local development agent.

**The Operator:** Agreed. Six containers meant six sets of logs to correlate when a tool call failed. However, maintaining a custom Squid configuration for outbound TLS interception is still fragile. If an Anthropic API endpoint changes, the agent dies silently. We need strict egress logging at the `Guard` layer so when a connection drops, we know exactly which domain was blocked.

**The Paranoid:** Complexity is the enemy of security, but over-simplification is worse. Keeping the `Scout` (Browser) and `Agent` (Brain) separated is non-negotiable. Modern Remote Browser Isolation (RBI) proves that the browser is the most vulnerable attack surface.  We must maintain that physical container gap to ensure untrusted web code never shares memory with the core reasoning engine.

## 2. The Security Theater

**The Paranoid:** Let's talk about container escapes. Dropping capabilities (`--cap-drop=ALL`) and applying a strict `seccomp` profile to block `unshare`, `mount`, and `bpf` syscalls is excellent. However, relying purely on capabilities is not enough; `security_opt: [no-new-privileges:true]` is crucial. Without it, a `setuid` binary inside the container can still escalate privileges and potentially exploit the kernel.

**The Pragmatist:** We also need to acknowledge the hardware reality. Because the stack runs on an Apple Silicon architecture, Docker is executing inside a lightweight Linux VM, not bare metal. A kernel exploit (like a cgroup `release_agent` escape) grants root access to the *VM*, not the host macOS environment.

**The Operator:** That's true, but it's bordering on security theater if the VM has highly sensitive directories mounted. If the `workspace` volume containing the `InvestmentToolkit` is mounted read-write, an attacker doesn't need to escape to the host—they just corrupt the project files, steal the valuation algorithms, or scrape API keys directly from the mounted volume.

## 3. The "Bus Factor" (Maintainability)

**The Operator:** Who maintains this when it breaks? The MVSA is readable. A single engineer can look at the four containers and understand the traffic flow: `Internet -> Nginx Guard -> Agent -> Proxy Sidecar -> Internet`.

**The Pragmatist:** Exactly. The previous iteration's attempt to intercept Chrome DevTools Protocol (CDP) commands between the Agent and the Scout was a maintenance nightmare. A simple network-level one-way firewall (`Agent -> Scout` allowed, `Scout -> Agent` denied) is much easier to maintain, test, and audit.

## 4. The Kill Chain

**The Paranoid:** Let's trace a realistic attack path. The agent is tasked with researching Broadcom (AVGO) financials using the Scout browser.

1. **Initial Compromise:** The Scout visits a compromised financial blog containing a zero-day Chromium exploit.
2. **Scout Takeover:** The attacker gains code execution inside the `Scout` container.
3. **Lateral Movement Attempt:** The attacker tries to reach the `Agent` container to access the `.env` files. *Blocked by the unidirectional Docker network policy.*
4. **Exfiltration Attempt:** The attacker tries to tunnel data out via DNS. *Blocked because the CoreDNS sidecar drops unwhitelisted queries.*

**The Pragmatist:** That covers the inbound web threat, but what about the agent's own autonomous actions? If the agent is instructed to update the `InvestmentToolkit` repository via its bash tool, its default behavior might be to execute a standard repository sync. We must ensure the system prompt and tool constraints strictly forbid executing a `git pull origin main`. Because the local main branch is intentionally kept behind the remote to protect incomplete specification work, a blind pull would set the project back to zero, causing catastrophic data loss without any "hacker" being involved at all. Operational guardrails are just as vital as network guardrails.

---

## 🏁 Final Verdict

### **Decision: A (Secure & Simple — Build It)**

The transition to the 4-Container MVSA (Guard, Agent, Scout, Unified Sidecar) successfully balances necessary isolation with maintainable complexity.

**Pre-Flight Checklist:**

1. Enforce `security_opt: [no-new-privileges:true]` on the Agent container in the Compose file.
2. Verify the unidirectional network policy separating the `Agent` and `Scout` containers is active.
3. Ensure the Agent's system prompt explicitly details branching strategies and forbids destructive Git commands on intentionally desynced local branches.
