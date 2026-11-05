# Cross-Provider Routing, Subscription Pooling, and Local Gateway Architecture

## 1. Problem Statement & Motivation

Autonomous multi-agent workflows (Claude Code, Antigravity `agy`, GitHub Copilot CLI, Codex) frequently encounter individual seat limits, rolling rate windows (e.g. 5-hour Claude windows, 3-hour ChatGPT caps), and API inference billing costs. Developers maintaining multiple commercial subscriptions (Claude Pro $20/mo, ChatGPT Plus $20/mo, Gemini Pro via AGY) often hit quota ceilings on one provider while others sit idle.

Solving this sustainably requires intelligent cross-provider fallback, subscription pooling, and local gateway proxying without violating operational or organizational boundaries.

---

## 2. Hard Policy & Workspace Boundary Governance

### The Separation Invariant: Work vs. Sovereign Research
* **GitHub Copilot (Work Strictly)**: Copilot access is provided for enterprise and work-related repositories only. **It must NEVER be routed or used for personal or sovereign research in Project Sanctuary.**
* **Personal Subscriptions (Sovereign Operations)**:
  - **Claude Pro ($20/mo)**: Claude Code CLI, architectural planning, code reviews, Sonnet/Opus reasoning.
  - **ChatGPT Plus ($20/mo)**: OpenAI Dots / Operator for GUI computer-use and browser tasks; OpenAI CLI endpoints.
  - **Gemini Pro (via AGY / Google One)**: Deep research dossiers, ultra-long context synthesis (1M-2M tokens), multimodal vision.

---

## 3. Proxy & Routing Architectures Evaluated

### Option A: Web-Scraping Session Proxies (e.g., 9Router / NymRouter) — REJECTED
* **Mechanism**: Proxies intercepting and hijacking browser session cookies and OAuth refresh tokens from web interfaces.
* **Failure Modes**:
  1. *Fragility*: Web session cookies expire unpredictably without refresh grants.
  2. *Security & Ban Risk*: Anti-bot heuristics detect automated headless REPL traffic and trigger account suspensions.
  3. *Protocol Corruption*: Lack of standard JSON schema translation silently drops or mangles Anthropic/OpenAI structured tool calls, corrupting file patches and multi-file diffs.
* **Verdict**: Strictly rejected for production agent engineering.

### Option B: Local Authenticated Gateway (CLIProxyAPI) — ADOPTED FOR LOCAL POOLING
* **Mechanism**: Open-source local proxy gateway listening on `localhost:4000`.
* **Capabilities**:
  - Intercepts standard OpenAI (`/v1/chat/completions`) and Anthropic (`/v1/messages`) requests from CLI agents.
  - Utilizes authentic CLI/OAuth credentials stored in local user harnesses.
  - Monitors rate budgets and rolling windows (5-hour Claude window, 3-hour GPT limits).
  - Clean tool-call translation ensuring zero git diff corruption.

### Option C: Enterprise Fallback Proxy (LiteLLM Proxy) — RECOMMENDED FOR COMPLIANCE
* **Mechanism**: High-reliability proxy translating requests across providers with strict YAML fallback chains.
* **Capabilities**:
  - Predictable cascading: Primary (Claude Pro) -> Fallback (Gemini Pro) -> Failsafe (Local Qwen2 / Ollama).
  - Zero brittle web scraping: Uses documented API / CLI protocols.
  - Full observability: Token telemetry, cost auditing, latency tracking.

---

## 4. Standardized Environment Fallback Configuration

CLI tools and subagents can target the local gateway via standard environment overrides:

```bash
# Point Claude Code / Anthropic SDK to local proxy
export ANTHROPIC_BASE_URL="http://localhost:4000"

# Point OpenAI SDK / CLI tools to local proxy
export OPENAI_BASE_URL="http://localhost:4000/v1"

# Failsafe local offline fallback (Protocol 128 compliant)
export LOCAL_INFERENCE_URL="http://localhost:11434/v1" # Local Ollama / Sanctuary-Qwen2
```

---

## 5. Decision Rules for Dispatch Evaluation

When evaluating tool or subagent dispatches (via `evaluating-agent-work`):
1. **Model Cost-Tier Matching**: Route repetitive boilerplate or test generation to cheaper models (Gemini Flash, Haiku) rather than frontier models (Opus, Astra).
2. **Pre-Delegation Falsifiability**: Define explicit success criteria, expected output paths, and timeout budgets before dispatch.
3. **Graceful Degrade**: If Claude Pro reaches the rolling cap, switch non-critical exploration to Gemini Pro or local models; pause state-changing commits until human partner review.
