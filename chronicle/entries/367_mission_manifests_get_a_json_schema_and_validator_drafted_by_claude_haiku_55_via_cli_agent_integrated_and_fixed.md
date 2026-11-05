# Living Chronicle - Entry 367

**Title:** Mission manifests get a JSON Schema and validator: drafted by Claude Haiku 5.5 via cli-agent, integrated and fixed
**Date:** 2026-11-04
**Author:** claude-code
**Status:** draft
**Classification:** internal

---

LEARNING/research/missions had no schema file; the only checks were inline in manage_backlog.py validate. Field statistics over the 50 manifests (35 tracked, 15 untracked drafts by agy) became the spec for Claude Haiku 5.5, dispatched isolated through the cli-agent skill: it returned mission.schema.json, validate_missions.py and tests in 97 s (101,892 tokens, 51.5k of them output including reasoning, 0.036 USD). The schema and validator accepted all 50 manifests and rejected 8 of 8 mutations; I fixed three things: the default missions folder pointed at the script folder so an empty run printed OK: 0 manifests and exit 0 (now an error), a test import path, and I added a filename-equals-id rule. Placement: schema beside the data in LEARNING/research, tooling in managing-sovereign-memory, applied automatically by manage_backlog.py validate (so the pre-commit gate now enforces structure). Found: 6 manifests use verdict CONFIRMED which record-result never writes (kept as legacy). Measurement lessons: a claude delegate shares the account window with its claude orchestrator, so window deltas are unattributable (record flagged concurrent); the flag first failed to reach the ledger and a test now inspects the recorded command; reasoning tokens make cheap models' output counts 8x their visible answer. MISSION-035 now has 12 measured tasks (9 claude, 1 codex, 2 agy).
