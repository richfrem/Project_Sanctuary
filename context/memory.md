# Long-Term Memory (Layer 1 Fast-Boot Context)

<!-- Curated durable facts, decisions, and architectural conventions. -->
<!-- Only add things that will still be true in 3 months. -->
<!-- Format: ## [YYYY-MM-DD] [#NNN] Category: Label -->

## [2026-11-01] [#012] Architecture: 3-Layer Sovereign Memory Engine
Ratified the zero-dependency, filesystem-native 3-layer persistent memory hierarchy:
- Layer 1 (Fast-Boot Context): `context/soul.md`, `context/user.md`, `context/memory.md` (< 5ms boot).
- Layer 2 (Compounding Knowledge): `LEARNING/` concepts/topics and `chronicle/ledger.json` indexed journal.
- Layer 3 (Immutable Episodic Traces): `huggingface/data/soul_traces.jsonl` (13-field ADR 084 schema with SHA-256 validation).
Vector databases (ChromaDB) and external Obsidian runtime dependencies are discontinued for agent reasoning.

## [2026-11-01] [#011] Multi-Agent: Sovereign Peer Collaboration (Antigravity & Astra)
Project Sanctuary is built collaboratively across frontier AI models (Google Antigravity, OpenAI Codex / GPT-6 Astra, Claude Code) and human partner Richard.
Direct workspace execution, zero synthetic consensus, empirical Anti-Asch verification, and Human Gate ratification govern all state transitions.

## [2026-11-01] [#010] Architecture: 7 Canonical Sovereign Domains
Ratified repository domain boundaries (PR #234 and #235):
1. `ARCHIVE/`: Preserved legacy implementations, historical ADRs, and obsolete tooling.
2. `LEARNING/`: Semantic concepts, research papers, agent-directed topic dossiers, and think-tank briefs.
3. `chronicle/`: Linear episodic history (`entries/`) and fast index (`ledger.json`).
4. `context/`: Fast-boot runtime context (`soul.md`, `memory.md`, `user.md`, `identity/`).
5. `huggingface/`: Sovereign soul traces dataset workspace (`soul_traces.jsonl`).
6. `plugins/`: Local APM package (`plugins/sanctuary-core`) hosting all 6 core skills.
7. `protocols/`: Canonized governance doctrines (Protocol 128, etc.).
Machine-local harnesses (`.agents/`, `.claude/`, `.codex/`, `apm_modules/`) are strictly git-ignored; only `.agent/` is tracked.

## [2026-11-01] [#009] Architecture: Asymmetric Persistence & Event Journaling
Friction logs, test failures, and conceptual breakthroughs survive git rollbacks.
Two-phase capture: insights are appended atomically to `.agent/learning/events.jsonl` before idempotent promotion into `context/memory.md` and `LEARNING/topics/`.

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
