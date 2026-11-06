# Responsibility map of `evaluating-agent-work` (verified 2026-11-06, MISSION-055)

Entry points: **ordinary work** (any repo, any task) -> the shared skill `evaluating-agent-outcomes`. **Sanctuary mission work** (manifests, verdicts, budgets, dashboards, Protocol 128 records) -> this skill. This skill routes generic evaluation to the shared implementation; it never re-implements it.

Classes: **G** generic (upstream, single implementation), **P** Sanctuary policy, **S** storage integration, **V** presentation.

| Script | Function / responsibility | Class | Where it lives now |
|---|---|---|---|
| `score_contribution.py` | rating validation, raw/fitness arithmetic, cost-adjusted ROI, bloat rule | G | upstream `scoring.py`, driven by `references/contribution-profile.json` |
| | projection fold / deterministic rebuild | G | upstream `scoring.fold_assessment` / `rebuild_projection` |
| | the five 0-3 dimensions and their meaning, fitness = raw/15, bloat thresholds | P | `references/contribution-profile.json` (+ `evaluation-rubric.md`) |
| | `events.jsonl` append (CONTRIBUTION_ASSESSED), `cognitive_ledger.json`, root discovery | S | here |
| | text/json output, CLI | V | here |
| `track_budget.py` | model cost tiers + lookup | G | upstream `accounting.py` (one table; the older divergent table that was unused by `main()` is gone) |
| | ledger fold (tasks, totals, 30 recent), turn check arithmetic | G | upstream `accounting.fold_usage` / `check_turns` |
| | subscription profiles (claude-pro, chatgpt-plus, gemini-pro) and soft limits | P | here (`BUDGET_PROFILES`) |
| | `budget_ledger.json`, `BUDGET_CONSUMED` events, CLI | S/V | here |
| `evaluate_dispatch.py` | model tier lookup | G | upstream `accounting.get_model_tier` |
| | Popperian pre-dispatch rules (falsifiable hypothesis, frontier only after repeated failures) | P | here |
| `usage_probe.py` | vendor usage parsers, snapshot/diff, codex/claude token readers, secret scrub, catalog prices, `cost_usd` | G | upstream `usage_measure.py` (moved verbatim; names re-exported for callers) |
| | static ceilings, floor policy, `headroom_report`, `recommend`, calibration, per-class predictor (falsified, MISSION-035) | P | here |
| | `load_measured` (reads this repo's budget events), CLI that records `BUDGET_CONSUMED` | S/V | here |
| `visualize_economy.py` | dashboards, RL simulation, Mermaid | V (reads Sanctuary ledgers) | here |
| `eval_gate.py` | validity/pilot/fixtures/select/interventions/repairs/invest | G | upstream `eval_core.py`; this file is a compatibility adapter (`--events`, `path=`, legacy `event_type`) |
| `sanctuary_adapter.py` | manifest -> work contract; delivery recorded apart from empirical state; shared validity gate interpreted against a mission (`empirical`) | P/S | here; no verdict-writing path |
| `counterexample_eval.py` | bounded seeded falsification search (`load_function`, `_execute`, `search`, `kill_worker`) | G (candidate) | **held here**, see below |
| | MISSION-041 benchmark protocol (`benchmark`, `summarize`, thresholds) | P | here |

## Held, with the concrete reason
`counterexample_eval.py`: its generic search core is a candidate for the shared skill, but MISSION-041's frozen v3 record (`LEARNING/experiments/mission-041/v3/freeze.json`) pins the SHA-256 of this file and its test, and the 041 harness imports it by path. Moving or editing it would make the recorded freeze unverifiable. It is therefore NOT duplicated upstream (one implementation exists, locally). Extracting it needs a decision to re-freeze or to record a superseding hash; until then the extraction is incomplete and documented as such.

## Verified compatibility
`tests/test_pre_migration_goldens.py` re-runs `golden_capture.py` against the refactored scripts and compares with outputs captured from the pre-migration code (commit 1f464bd52): model cost weights and tiers for 35 names, score/fitness/ROI/bloat for 30 rating-model pairs, the real `score_contribution` CLI (stdout, event line, ledger), projection rebuild, the `track_budget` ledger after three records, both `check` results, and `cost_usd`. All identical; historical scores and ledgers keep their meaning. One intentional change: the unused older table in `track_budget.get_model_tier` (it mis-tiered e.g. `claude-opus-5-5` as standard) now returns the shared tier; nothing called it.

## Ordinary work vs mission work
Ordinary work: `outcome_eval.py assess` (work contract) with an optional rubric (`scoring.py`). Mission work: `sanctuary_adapter.py contract|deliver|state|empirical`, then `manage_backlog.py record-result` only on valid evidence and the manifest's own falsification criteria.
