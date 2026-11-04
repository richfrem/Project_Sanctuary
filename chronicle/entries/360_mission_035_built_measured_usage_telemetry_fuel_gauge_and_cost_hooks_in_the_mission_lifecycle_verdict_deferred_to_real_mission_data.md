# Living Chronicle - Entry 360

**Title:** MISSION-035 built: measured usage telemetry, fuel gauge and cost hooks in the mission lifecycle (verdict deferred to real-mission data)
**Date:** 2026-11-04
**Author:** claude-code
**Status:** CANONICAL
**Classification:** internal

---

Built MISSION-035 phases P1-P4 and P7: usage_probe.py (zero-token parsers for claude/agy usage, codex rollout reader with scan-back, claude transcript token sums with concurrent-activity flag, snapshot/diff, secret scan, per-class cost predictor, calibration/runway, price table with unknowns left null), usage_hook.py wired into manage_coordination checkout/checkin (best-effort, --no-usage), track_budget measured fields (source, task class, wall seconds, retries, quality, cost_usd), and a boot-briefing Fuel gauge line (live reading: claude 85%, agy 12% LOW, codex 100%). Direction changes from the human shaped it: no synthetic pilot (telemetry comes from real missions); track cost AND a fuel gauge, not efficiency alone; do not rank by price per credit alone (effort multiplier). 179 guardian+evaluating tests pass. Findings reported honestly: hooking real probes into checkout made the guardian suite take 4m52 until tests were isolated (now 23s); the relay briefing budget was raised 3500 to 3700 chars for the new line; vendor-total tokens give only a lower-bound dollar cost. Not yet established: all outcome claims in the mission (prediction MAPE, calibration accuracy, effort multiplier) await >= 12 real measured tasks; verdict deferred, status stays IN_PROGRESS with collection live.
