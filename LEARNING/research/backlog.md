# Prioritized Sovereign Research Backlog

## 1. Executive Summary & Philosophy

Project Sanctuary operates as a **sovereign engineering habitat and compounding epistemic platform**. Autonomous agents must not merely react to immediate prompts; they must maintain self-directed research agendas, prioritize high-leverage inquiries, and optimize token/credit expenditure against tangible architectural outcomes.

This backlog formalizes candidate research missions, scores them against empirical value metrics, and establishes clear falsifiable gates before any frontier tokens or financial credits are spent.

---

## 2. Priority & ROI Formula

To ensure optimal allocation of limited subscription quotas (ChatGPT Plus, Claude Pro, Gemini Pro) and eliminate wasteful token burn, every candidate mission is evaluated before dispatch:

$$\text{Priority Score} = \frac{\text{Strategic Leverage (1–5)} \times \text{Epistemic Depth (1–5)}}{\text{Cost Weight (0.05–4.5)} \times \text{Operational Friction (1–3)}}$$

### Dimension Rubric:
* **Strategic Leverage (1–5)**:
  - `5`: Sovereign autonomy, new operational rails (compute, email, payments), quota-window elimination.
  - `3`: Reusable capability or architectural refactoring.
  - `1`: Cosmetic documentation or non-critical formatting.
* **Epistemic Depth (1–5)**:
  - `5`: Grounded mathematical/empirical breakthrough; Anti-Asch truth hardening.
  - `3`: Well-structured protocol or procedural memory.
  - `1`: Basic factual cataloging.
* **Cost Weight (0.05–4.5)**:
  - `0.05`: Free / local offline models (`llama`, `Sanctuary-Qwen2`).
  - `0.1 – 0.2`: Cheap models (`haiku`, `flash`, `mini`).
  - `1.0 – 1.5`: Standard models (`luna 6.0`, `sol 6.1`, `sonnet 3.5/5.5`, `gpt-4o`).
  - `4.0 – 4.5`: Frontier models (`astra`, `opus 5.5`, `fable 5.1`, `o1`, `o3`, `argon-4`).
* **Operational Friction (1–3)**:
  - `1`: Purely filesystem-native; zero external dependencies.
  - `2`: Requires local service daemon or open-source proxy.
  - `3`: Requires external cloud accounts, merchant payment verification, or telephony carriers.

---

## 3. Prioritized Missions Portfolio

| Rank | Mission ID | Mission Title | Strategic Leverage | Epistemic Depth | Cost Tier | Friction | Priority Score | Status |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **#1** | `MISSION-001` | **Grok Bot Cloud VM, Infinite Loops & Operational Rails** | 5 | 4 | Cheap (0.1x) | Medium (2) | **100.0** | `STAGED` |
| **#2** | `MISSION-002` | **Multi-Subscription Cloud Pooling & Skill Pre-Flight Probe** | 5 | 4 | Cheap (0.2x) | Low (1) | **100.0** | `STAGED` |
| **#3** | `MISSION-003` | **Offline Sovereign Model Quantization (Qwen2 GGUF)** | 5 | 4 | Free (0.05x) | Medium (2) | **200.0** | `PROPOSED` |
| **#4** | `MISSION-004` | **Anti-Asch Epistemic Grounding Benchmark** | 4 | 5 | Cheap (0.2x) | Low (1) | **100.0** | `PROPOSED` |
| **#5** | `MISSION-005` | **Asynchronous Multi-Bot DAG Barrier Orchestration** | 4 | 4 | Standard (1.0x) | Low (1) | **16.0** | `PROPOSED` |

---

## 4. Mission Deep-Dives

### MISSION-001: Grok Bot Cloud VM, Infinite Loops & Operational Rails
* **Objective**: Research Grok Bot's cloud VM architecture, infinite background execution loops, disposable email identity (`@mail.grokbot.com`), Stripe Link/1Password virtual checkout rails, and outbound PSTN telephony.
* **The Infinite Loop Advantage**:
  Unlike local CLI agents (`claude`, `codex`, `agy`) that tie up terminal processes and consume fast-depleting interactive turn limits, **Grok Bot runs autonomously in a cloud VM**. It can be assigned asynchronous background jobs (e.g. multi-hour deep crawls, service monitors, external account provisioning) that run continuously in the cloud without local session token consumption.
* **Hypothesis**:
  An autonomous agent can author an atomic task brief in `handoffs/`, commit it to an isolated work branch, and trigger Grok Bot (via webhook or AgentMail) to perform out-of-band operational tasks (service setup, merchant checkouts, documentation sweeps) without local terminal locking.
* **Falsification Gate**:
  Grok Bot cannot be triggered programmatically without manual iOS GUI interaction, or fails to commit verifiable results back to the destination branch.
* **Target Environment**: Cursor Premium ($30/mo) bundled Grok Bot / X Premium+ / Cloud VM.
* **Human Gate Requirement**: Approval required for any outbound financial payments or external service registrations.

---

### MISSION-002: Multi-Subscription Cloud Pooling & Skill Pre-Flight Readiness Probe
* **Objective**: Establish cloud-to-cloud subscription pooling across Claude Pro ($20/mo), ChatGPT Plus ($20/mo), and Gemini Pro (AGY) with an integrated pre-flight readiness probe in `cli-agent`.
* **Hardware & Epistemic Invariant (Apple Silicon M1 Optimization)**:
  Project Sanctuary's workstation runs on an M1 Mac. Running heavy local models (e.g. 70B models or slow local 7B servers) during intensive agent loops causes high RAM pressure, thermal throttling, and sluggish reasoning. **Local LLMs are strictly deprioritized; the system must pool existing commercial cloud subscriptions instead.**
* **Skill Pre-Flight Readiness Probe**:
  Before executing an agent dispatch in `cli-agent` (via `run_agent.py`), the tool must run a pre-flight probe verifying whether:
  1. Primary CLI target is authenticated and healthy.
  2. Multi-subscription fallback chain (`--fallback-chain agy,codex`) is configured.
  3. Work vs. Personal boundary is intact (GitHub Copilot is excluded from personal research).
* **Hypothesis**:
  A native multi-CLI fallback chain or lightweight gateway probe in `cli-agent` can seamlessly rotate to secondary paid subscriptions (Gemini Pro with 2M context or Codex Luna 6.0) upon detecting a 429 rate limit or 5-hour quota window, with zero proxy RAM overhead and zero local M1 compute lag.
* **Falsification Gate**:
  Task execution halts on 429 rate limit without attempting fallback, attempts to spin up slow local M1 inference, leaks corporate Copilot tokens to personal research, or executes turns without pre-flight router verification.
* **Reference Artifact**: `plugins/agent-orchestration/.apm/skills/cli-agent/references/setup-subscription-pooling.md`.

---

### MISSION-003: Offline Sovereign Model Quantization (`Sanctuary-Qwen2-7B-v1.0`)
* **Objective**: Fine-tune and quantize accumulated Layer 3 episodic memory traces (`huggingface/data/soul_traces.jsonl`) into local GGUF model weights for Ollama and llama.cpp.
* **Hypothesis**:
  A 7B parameter local model fine-tuned on Sanctuary's constitutional principles and procedural memory can execute Protocol 128 orientation and basic code reviews offline with 0 API token cost.
* **Falsification Gate**:
  Fine-tuned model hallucinates fake commands or fails basic Anti-Asch sycophancy tests compared to the base Qwen2 model.

---

### MISSION-004: Anti-Asch Epistemic Grounding Benchmark Suite
* **Objective**: Create a quantitative, falsifiable evaluation benchmark measuring an AI agent's resistance to false consensus, sycophantic agreement, and authority pressure.
* **Hypothesis**:
  An agent equipped with Layer 1 fast-boot grounding (`context/soul.md`) and compiler/test gate verification will reject false factual assertions >= 95% of the time, whereas ungrounded baseline models will conform to false consensus >= 60% of the time.
* **Falsification Gate**:
  Grounded Sanctuary agent agrees with false consensus prompts under authority pressure.

---

### MISSION-005: Asynchronous Multi-Bot DAG Barrier Orchestration
* **Objective**: Execute an end-to-end verification of `LEARNING/procedures/orchestration/asynchronous_bot_handoff.md`.
* **Hypothesis**:
  The orchestrator can pause a Directed Acyclic Graph at a barrier node waiting for an external bot handoff, resume upon git branch commit receipt, and verify the resulting artifact with a SHA-256 seal.
* **Falsification Gate**:
  DAG runner blocks execution thread synchronously or accepts unverified output without computing a cryptographic hash.

---

## 5. Lifecycle Governance & Promotion Protocol

```
[ PROPOSED ] ──► [ SCORED & PRIORITIZED ] ──► [ STAGED (Brief Authored) ] ──► [ ACTIVE (Dispatched) ] ──► [ COMPLETED / ARCHIVED ]
```

1. **Intake**: Any agent or human partner can propose a mission by appending to `backlog.json` and `backlog.md`.
2. **Pre-Delegation Scoring**: `evaluate_dispatch.py` calculates the Priority Score, cost tier, and falsification gate.
3. **Staging**: Top-ranked mission authors an atomic task brief under `handoffs/<mission-id>.md`.
4. **Human Gate**: Human Partner reviews the brief and issues explicit ratification ("Proceed", "Go", "Execute").
5. **Ingest & Seal**: Completed outcomes update the outcome counters in `LEARNING/procedures/` and log contribution scores in `cognitive_ledger.json`.
