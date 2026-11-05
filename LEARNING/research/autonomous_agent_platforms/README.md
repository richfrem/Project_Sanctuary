# Autonomous Agent Platforms & Cross-Provider Routing Dossier

**Domain**: Autonomous AI Agent Architectures & Delegation Topologies  
**Status**: ACTIVE / CONFIRMED  
**Date**: November 2, 2026  
**Authors**: Richard (Human Steward), Antigravity (Gemini Sovereign Pair Agent), Astra (GPT-6 Astra)

---

## 1. Executive Summary

As multi-agent software engineering matures into 2026, autonomous agent architectures have bifurcated into two distinct operational paradigms:

1. **Synchronous Developer Coding CLIs**:
   * Tools like Claude Code CLI, Codex CLI, Antigravity (`agy`), Copilot CLI, and `cursor-agent`.
   * These operate directly on local POSIX filesystems via standard input/output (`stdin`/`stdout`) streams, local file locks, and git worktrees.
   * Bound by strict subscription rate windows (Claude 5-hour rolling caps, ChatGPT 3-hour caps).

2. **Asynchronous Commercial Agent VMs & Operators**:
   * Platforms like **xAI Grok Bot**, **Anthropic Claude Duet**, **OpenAI Dots (Operator)**, and **Google Gemini Agent (Project Astra)**.
   * These run in isolated cloud/desktop VMs with native web, GUI vision, telephony, and SaaS connectors.
   * They do not expose local stdio REPLs and cannot be wrapped in shell subprocesses.

This dossier codifies the integration boundaries, delegation topologies, and multi-subscription pooling architectures enabling Project Sanctuary to collaborate effectively across both worlds while conserving real-world subscription quotas.

---

## 2. Dossier Documents

* **[Platform Capability Matrix](./platform_capability_matrix.md)**:
  Exhaustive taxonomy comparing Grok Bot, Claude Duet, OpenAI Dots/Operator, and Gemini Agent across execution boundaries, identity/auth, core differentiators, and integration contracts.
* **[Cross-Provider Routing & Subscription Pooling](./cross_provider_pooling.md)**:
  Technical specification for local proxy gateways (`localhost:4000`), failover mechanics, and strict workspace boundary governance (work vs personal/sovereign accounts).

---

## 3. Core Architectural Decisions

1. **Never Wrap Commercial VMs in Synchronous Stdio Pipes**:
   Commercial agent VMs like Grok Bot must be engaged via **Asynchronous Git & Webhook Handoffs** (committing task briefs to feature branches or triggering webhooks), preserving clean execution boundaries.
2. **Work vs. Sovereign Project Isolation**:
   GitHub Copilot and enterprise seats are strictly isolated to employer/work-related repositories. Sovereign and open-source research is powered by personal subscriptions (Claude Pro, ChatGPT Plus, Gemini Pro) and local offline LLMs (`llama-server`).
3. **Local Gateway Proxy Failover**:
   CLI agent traffic is routed through local proxies (`LiteLLM` / `CLIProxyAPI`) on `localhost:4000` to seamlessly absorb 429 rate limits without crashing unattended agent loops.
