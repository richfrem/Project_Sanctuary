# MISSION-035 results at N = 14 measured tasks (2026-11-05)

Collected passively during real missions and delegated tasks (11 claude, 1 codex, 2 agy; 9 free-text task classes; all quality-scored). Commands: `usage_probe.py calibrate`, `usage_probe.py evaluate`, plus a leave-one-out check I ran for criterion (f). Verdict recorded: **FALSIFIED** (as pre-registered; see "What survived").

## Scorecard against the pre-registered criteria
| | Criterion | Result |
|---|---|---|
| (a) | exact per-task tokens for >= 3 of 4 backends, within +/-10% of vendor totals | **Met** for claude, codex, agy. Claude cost computed $29.49 vs vendor $29.55 (0.2%); Codex rollout total equals the 27,943 it reported; agy tokens are its own JSON `usage`. Local has no quota or token surface. |
| (b) | window readings flagged noisy when other sessions are active | **Partly met**: the flag works inside one project folder; it cannot see a session in another folder (found when a Haiku delegate's window delta included the orchestrator). Delegates are now flagged by rule. |
| (c) | held-out token MAPE <= 30% and >= 20% below static ceilings | **Not met**: learned **122.3%** vs static 166.2% (improvement 26.4% passes the relative leg; the 30% limit fails by 4x). n_train 9, n_held_out 5, across 9 task classes with 1-4 tasks each. |
| (d) | measured effort multiplier per model and task class | **Untestable**: no task class has been run on two models. |
| (e) | recommender beats a random choice on held-out tasks | **Not tested.** |
| (f) | calibrated percent-to-token within +/-50% on held-out measurements | **At the falsification line**: `calibrate` = 56,349 tokens per percent (about 5.63M per 5-hour window; 3.6M left at 64%). Leave-one-out on the 9 usable non-concurrent claude records: **5 of 8 within +/-50%**; the 3 misses (57%, 58%, 126%) equal the threshold "more than 50% off on >= 3". |

## Why the per-class predictor failed (what the data shows)
- 14 tasks over 9 free-text classes (`build`, `fix`, `refactor`, `feature`, `measurement`, `housekeeping`, `drafting`, `inspection`, `test-writing`): a per-class median is fitted from 1-3 points, so it is noise. The static ceilings are also bad (166%), so the task sizes themselves vary by 10x inside a class.
- Orchestrator token counts are dominated by long-lived context (cache writes and reads), not by the task: the schema task cost Haiku $0.036 and the orchestrator around it $5.59.

## Why (f) is borderline, not a clean failure
The claude 5-hour reading is an integer percent. A task that moved it "2" really moved 1.5-2.5, which alone is 25-50% error for small tasks. All 3 misses are 1-2 percent tasks; the one large task (333,934 tokens, 6%) was predicted within 5%. The pre-registered letter still counts them.

## Findings that stand (independent of the verdict)
- agy `--output-format json` reports exact per-call usage; a trivial call carries about 18.5k input tokens of overhead.
- Reasoning tokens bill as output: Haiku 5.5 returned about 6k visible tokens and billed 51.5k output tokens.
- A claude delegate shares the 5-hour window with its claude orchestrator, so its window delta is unattributable.
- Two agy runs: 3.7x more billable tokens moved the 5-hour window 2.2x as much (cached reads differed), so quota is not proportional to billable tokens.
- Cost computed from the cli-agent catalog is accurate when the model id is exact; `claude-haiku-5-5` and `gemini-3.8-flash-low` priced correctly.

## What survived
Exact per-task tokens for claude/codex/agy; cost from the cli-agent catalog; boot fuel gauge and checkout headroom warnings; `usage_hook.py delegated` (agy/codex/claude delegates); delegate flagging; the `recommend` and `calibrate` tools (usable with the stated error bars). What did not: a per-free-text-class token predictor at this sample size and a percent calibration good enough to call within +/-50%.

## Follow-up
MISSION-051 tests the specific repairs this data points to: few agent-declared size buckets instead of free-text classes, percent calibration only where a window moved enough to read (or pooled), and at least one same-bucket run on two models for the effort multiplier.
