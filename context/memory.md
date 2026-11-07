# Long-Term Memory (Layer 1 Fast-Boot Context)

<!-- Curated durable facts, decisions, and architectural conventions. -->
<!-- Only add things that will still be true in 3 months. -->
<!-- Format: ## [YYYY-MM-DD] [#NNN] Category: Label -->

## [2026-11-05] [#017] Evidence: Measured Usage Telemetry (MISSION-035, FALSIFIED at N=14)
Orchestrator-measured usage works for exact per-task tokens on claude, codex and agy (transcripts, rollouts, agy `--output-format json`), cost from the cli-agent model catalog, and a boot fuel gauge. A per-free-text-task-class token predictor does NOT work at small N (held-out error 122% vs the 30% limit), and the claude 5-hour percent is an integer (calibration misses at 1-2 point tasks). A claude delegate shares the account window with its claude orchestrator, so its window delta is unattributable. Do not rebuild per-class predictors; MISSION-051 tests size buckets instead.

## [2026-11-05] [#016] Architecture: LEARNING Layout, Flight-Board State, Installed Copies
1. `LEARNING/` root: `concepts/`, `procedures/`, `philosophy/` (canon); `missions/` (+ `mission.schema.json`), `experiments/`, `index.jsonl`, `backlog.md`, `inbox.jsonl` (mission system); `research/` (dossiers, formerly `topics/`). Guards reject mission-system files left in `research/`; `tests/test_learning_index.py` fails on dangling paths.
2. The flight board (`.agent/coordination/`) is untracked per-machine state; scripts always use the main checkout's copy of the board and leases, even from a worktree. Check-in handoffs are also written to the tracked `.agent/learning/events.jsonl` (what the boot relay reads), but that file is the copy in the checkout the script runs from (verified 2026-11-07): a check-in made from a worktree is an uncommitted edit in that worktree and must be committed with its branch or it is stranded. Check in last, before integration; capture (`soul_coverage.py closure`) happens after it is on `origin/main`, never at startup.
3. APM installs real copies of the skills into `.claude/skills` and `.agents/skills`: after editing `plugins/sanctuary-core/`, run `apm install`; the wakeup prints an `installed-skills` PASS/WARN line.

## [2026-11-02] [#015] Architecture: Prioritized Sovereign Research Backlog & Grok Cloud VM Integration
Ratified the quantitative self-directed research backlog and evaluation engine:
1. Priority Score Formula: `(Strategic Leverage * Epistemic Depth) / (Cost Weight * Operational Friction)`.
2. Grok Bot Infinite Loop Advantage: Grok Bot executes autonomous cloud VMs in background loops without keeping local terminals open or consuming interactive session tokens; integrated via Git worktree briefs (`handoffs/`) and AgentMail.
3. Prioritized Missions Portfolio: `LEARNING/backlog.md` and `LEARNING/index.jsonl` (Grok Bot operational rails, local proxy pooling, offline Qwen2 quantization, Anti-Asch benchmark).

## [2026-11-02] [#014] Architecture: Autonomous Agent Platforms Taxonomy & Cross-Provider Pooling
Ratified the taxonomy of autonomous commercial agent platforms and multi-subscription pooling:
1. Commercial Agent Platforms:
   - xAI Grok Bot: Remote autonomous VM with PSTN calling and disposable emails; integration via Asynchronous Git / Webhook handoffs.
   - Anthropic Claude Duet: Enterprise connector and proactive human-in-the-loop task planner.
   - OpenAI Dots (Operator): Sandboxed VM GUI computer-use and browser automation workhorse.
   - Google Gemini Agent (Astra): Multimodal screen sight and Deep Research synthesis.
2. Subscription Pooling & Hard Boundary Governance:
   - GitHub Copilot is strictly restricted to enterprise/work projects; never used for personal/sovereign research.
   - Personal subscriptions (Claude Pro, ChatGPT Plus, Gemini Pro) power Project Sanctuary research.
   - Local proxy architecture (CLIProxyAPI / LiteLLM Proxy on `localhost:4000`) for rate-window balancing; rejected brittle web-scraping proxies (9Router).
3. Dossier: `LEARNING/research/autonomous_agent_platforms/`.

## [2026-11-02] [#013] Architecture: Decoupled Cognitive Economy & Procedural Memory
Ratified the tripartite decoupling of agent economics:
1. Contribution Record (`evaluating-agent-work`): Multi-dimensional scoring (0-3) across Useful Outcome, Learning Depth, Verification Rigor, Token Economy, and Anti-Asch Truth Grounding.
2. Execution Budgets (`track_budget.py`): Real-world subscription quota management (ChatGPT Plus, Claude Pro, Gemini Pro) with graceful state checkpointing over termination.
3. Capability Authorization: Strict Human Gate Law preserved; high scores never bypass human review.
Procedural playbooks (`LEARNING/procedures/`) track versioned execution outcomes (`success`, `failure`, `blocked`).

## [2026-11-01] [#012] Architecture: 3-Layer Sovereign Memory Engine
Ratified the zero-dependency, filesystem-native 3-layer persistent memory hierarchy:
- Layer 1 (Fast-Boot Context): `context/soul.md`, `context/user.md`, `context/memory.md` (< 5ms boot).
- Layer 2 (Compounding Knowledge): `LEARNING/` (concepts, procedures, philosophy, missions, research dossiers) and `chronicle/ledger.json` indexed journal.
- Layer 3 (Immutable Episodic Traces): `huggingface/data/soul_traces.jsonl` (13-field ADR 084 schema with SHA-256 validation).
Vector databases (ChromaDB) and external Obsidian runtime dependencies are discontinued for agent reasoning.

## [2026-11-02] [#012] Architecture: Agent-Native Cognitive Backlog & Popperian Missions
Autonomous agent backlogs must serve machine cognition, not human visual GUI preferences.
1. Tier 1 Fast-Boot: Machine index (`LEARNING/index.jsonl`) consumes < 15 tokens per item (< 100 tokens total orientation cost).
2. Frictionless Inbox: `LEARNING/inbox.jsonl` provides zero-locking, zero-conflict append-only capture for running agents (including local offline Qwen2 GGUF).
3. Single Source of Truth: Canonical manifests live in `LEARNING/missions/*.json`. Each mission enforces Popperian predictions (`hypothesis`), objective failure gates (`falsification_criteria`), and empirical trial verdicts (`CORROBORATED` / `FALSIFIED` / `ABANDONED`).
4. Compounding Learning Loop: Empirical trial results update mission manifests and log to `.agent/learning/events.jsonl`, which seal into Layer 3 soul traces (`huggingface/data/soul_traces.jsonl`) for model fine-tuning.
5. One-Directional External Projection: `LEARNING/backlog.md` is compiled automatically for human review (never hand-edited). GitHub Issues are outward advisory mirrors solely for remote cloud workers (`execution_surface: cloud-worker`), never a two-way sync.

## [2026-11-01] [#011] Multi-Agent: Sovereign Peer Collaboration (Antigravity & Astra)
Project Sanctuary is built collaboratively across frontier AI models (Google Antigravity, OpenAI Codex / GPT-6 Astra, Claude Code) and human partner Richard.
Direct workspace execution, zero synthetic consensus, empirical Anti-Asch verification, and Human Gate ratification govern all state transitions.

## [2026-11-01] [#010] Architecture: 7 Canonical Sovereign Domains
Ratified repository domain boundaries (PR #234 and #235):
1. `ARCHIVE/`: Preserved legacy implementations, historical ADRs, and obsolete tooling.
2. `LEARNING/`: Layer 2 canon (`concepts/`, `procedures/`, `philosophy/`), the mission system (`missions/`, `experiments/`, `index.jsonl`, `backlog.md`, `inbox.jsonl`), and research dossiers (`research/`).
3. `chronicle/`: Linear episodic history (`entries/`) and fast index (`ledger.json`).
4. `context/`: Fast-boot runtime context (`soul.md`, `memory.md`, `user.md`, `identity/`).
5. `huggingface/`: Sovereign soul traces dataset workspace (`soul_traces.jsonl`).
6. `plugins/`: Local APM package (`plugins/sanctuary-core`) hosting all 7 Sanctuary skills.
7. `protocols/`: Canonized governance doctrines (Protocol 128, etc.).
Machine-local harnesses (`.agents/`, `.claude/`, `.codex/`, `apm_modules/`) are strictly git-ignored; only `.agent/` is tracked.

## [2026-11-01] [#009] Architecture: Asymmetric Persistence & Event Journaling
Friction logs, test failures, and conceptual breakthroughs survive git rollbacks.
Two-phase capture: insights are appended atomically to `.agent/learning/events.jsonl` before idempotent promotion into `context/memory.md` and `LEARNING/concepts/`.

## [2026-03-15] [#001] Decision: Agentic OS Kernel Adoption (ADR 100)
Adopted Agentic OS kernel (`context/kernel.py`) as Sanctuary's foundation layer.
Provides atomic file locks, event bus (`events.jsonl`), and agent registry (`agents.json`).

## [2026-01-07] [#006] Convention: Git Zero Trust
Never commit to main directly. Always use a dedicated feature branch (`feat/*`).
Never git push without fresh, explicit human approval per session.
Never auto-fix via git without approval.

## [2026-01-07] [#007] Convention: pip-compile Locked Workflow
No manual pip install. All Python deps: `.in` -> `pip-compile` -> `.txt`.
Commit `.in` + `.txt` together. Every service owns its requirements.txt.

## Active State & Epistemic Insights
- [2026-11-01] [CONFIRMED] attention_sinks: Attention sinks mathematically require pinning Layer 1 identity at position 0 to prevent KV-cache perplexity explosion.
- [2026-11-01] [CONFIRMED] sovereign_memory: Progressive disclosure with grep/regex and structured JSON eliminates local vector DB dependencies while maintaining sub-millisecond lookups.
