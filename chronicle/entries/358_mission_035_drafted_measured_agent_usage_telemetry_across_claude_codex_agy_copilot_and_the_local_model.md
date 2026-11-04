# Living Chronicle - Entry 358

**Title:** MISSION-035 drafted: measured agent usage telemetry across claude, codex, agy, copilot and the local model
**Date:** 2026-11-04
**Author:** claude-code
**Status:** PROPOSED
**Classification:** internal

---

Drafted MISSION-035 (priority 13.33): an orchestrator-run before/after usage probe feeding evidence-based model assignment in evaluating-agent-work. Headless probes established before drafting: 'claude -p /usage' and 'agy -p /usage --output-format json' return 5-hour and weekly window state at 0 tokens; codex has no headless usage command ('/status' under codex exec is sent to the model and cost 18,369 tokens) so windows come from rollout rate_limits and its interactive /usage dashboard (relative units, up to 6h stale; Chats tab gives per-chat weekly and 5-hour percent); copilot shows plan AIC (2,388/6,000) and has --usage-output-file, gated as work-only. Costs of the research, reported honestly: an 18,369-token codex probe, and a blind digit sequence while driving the codex TUI reached the 'Use this reset?' confirmation (backed out unconfirmed; config unchanged), now a read-only TUI-driving constraint in the mission. Live constraint found: agy Gemini weekly quota 12% remaining. Process miss: PR 297 was opened but not merged and the worktree left open until the human pointed it out; the full cycle (merge, pull, cleanup, resync) was then completed.
