# Living Chronicle - Entry 361

**Title:** Companion repo catalog refresh (Protocol 133): Claude 5.5, GPT-6 family, Gemini 4 Argon; exact-model telemetry costs validated
**Date:** 2026-11-04
**Author:** claude-code
**Status:** CANONICAL
**Classification:** internal

---

Executed the first real Protocol 133 change under the human's explicit direction. cli-agent model catalogs (verified 2026-09-09) lacked the models in use. Refreshed claude/codex/agy catalogs in the companion clone from official pricing pages plus the live agy picker and Codex CLI model cache; fixed select_model raising TypeError on null prices (2 tests, pre-existing for agy low tier); updated stale picker-snapshot tests; recorded map-debt (2 OPEN items: sync_cheapest_models.py path expects non-.apm layout; run_agent prompt-echo test fails on main, pre-existing). Merged as companion PR 747. Cross-check: a real Claude Code session on claude-sonnet-5-5 reported $29.55 and the refreshed catalog prices it at $29.49 (1h cache-write rate); the older sonnet-5 stand-in overestimated by 52%. Deployment finding: 'apm install --global' did NOT update ~/.claude/skills because the GitHub dependency richfrem/agent-plugins-skills was pinned to an older commit in the lockfile; 'apm update --global' moved all 7 packages (hundreds of files) from 3f0c0ba to 6bb64dc, so Protocol 133's install step needs the update command (and one interrupted-install backup item could not be removed). Sanctuary side: usage_probe now prices 1h cache writes and measures codex per-task tokens automatically.
