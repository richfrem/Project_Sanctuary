# Living Chronicle - Entry 369

**Title:** MISSION-035 closed FALSIFIED at 14 measured tasks; what survived; MISSION-051 follow-up
**Date:** 2026-11-04
**Author:** claude-code
**Status:** draft
**Classification:** internal

---

At 14 measured tasks (11 claude, 1 codex, 2 agy; 9 free-text task classes) the pre-registered checks ran. Per-class token predictor: held-out MAPE 122.3% vs 166.2% for the static ceilings (limit 30%; relative improvement 26.4% passes its 20% leg). Percent-to-token calibration: 56,349 tokens per percent, about 5.63M per 5-hour window; leave-one-out 5 of 8 within +/-50%, and the 3 misses equal the falsification threshold (all 1-2 percent tasks where an integer reading alone is 25-50% error; the one large task was predicted within 5%). Exact per-task tokens held for claude, codex and agy (3 of 4 backends); the effort multiplier and the recommender could not be tested because no task class ran on two models. Recorded FALSIFIED as pre-registered, with what survived listed: exact tokens, catalog-based cost, fuel gauge, delegation recorder. Follow-up MISSION-051 (score 30) tests size buckets instead of free-text classes, calibration only where the window moved at least 3 points, and a same-bucket two-model effort ratio, with definitions fixed in advance; linked both ways to 035, 016 and 045. Findings: orchestration cost dominates delegation cost (Haiku draft 0.04 USD vs 5.59 USD of orchestrator work around it); reasoning tokens bill as output; a claude delegate shares the account window with its claude orchestrator. The human chose to record the verdict now rather than keep collecting.
