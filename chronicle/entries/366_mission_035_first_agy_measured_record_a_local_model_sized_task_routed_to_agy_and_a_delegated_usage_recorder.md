# Living Chronicle - Entry 366

**Title:** MISSION-035: first agy measured record (a local-model-sized task routed to agy) and a delegated-usage recorder
**Date:** 2026-11-04
**Author:** claude-code
**Status:** draft
**Classification:** internal

---

On the user's suggestion a task the router sends to free-local qwen (compare MISSION-004 and MISSION-009) was routed to agy (gemini-3.8-flash-low). The text-mode dispatch answered in 10 s but reports no usage, so its tokens are unmeasured. agy --output-format json does report usage, so the task was repeated in JSON mode: 5,510 input, 502 output, 33,091 cache-read, 6,012 total, 0.0085 USD from the cli-agent catalog; the Gemini 5-hour window moved 0.028 points and the weekly 0.008. agy's answer was checked against both manifests: the two missions are duplicates (same thresholds, different wording and tags). Not applied: closing MISSION-009 is the human's call. New: usage_probe.parse_agy_json_usage, usage_hook.finish token_split, and a usage_hook.py delegated command that records delegated agy or codex work in one step; the sources reference documents it. Finding: a trivial agy call carries about 18.5k input tokens of fixed overhead. MISSION-035 now has 9 measured tasks (7 claude, 1 codex, 1 agy).
