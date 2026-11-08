# MISSION-060 execution handoff: claude-code to codex

**Written:** 2026-11-07 by claude-code. **Mission state:** STAGED (authorized by Richard's "Proceed" on 2026-11-07), assignee codex, **not started**. Executor: GPT-6.1 Sol (`gpt-6.1-sol`) at medium effort, confirmed by Richard on 2026-11-07 ("proceed assign to sol 6.1"). **Zero model calls have been made for this study; zero study tokens spent.**
Read first: `LEARNING/experiments/mission-060/exploration-policy-revision.md` (why and what), then this file. The mission manifest is `LEARNING/missions/MISSION-060.json`; the superseded original design is preserved in its `revision_history`.

## 1. What was approved (the package Richard approved with "Proceed")
- Protected share 30% of the study's 16 selection slots (11 formula + 5 interest-led). Adopting a standing share in backlog tooling is NOT approved and is NOT part of this mission.
- Thresholds (frozen in `thresholds.json`): B1 = P-only outputs per budget at least 1.5x F-only AND at least 2 more outputs; B3 guard 90%; claim A unchanged; at least 4 items per cohort or UNASSESSED.
- Budget: ceiling 700,000 additional model tokens; 25,000 per item; at most 10 executed items (5 + 5). Hugging Face sync is NOT authorized (add it only if Richard says so).
- Clients as approved: declarers Claude Sonnet 5.5, AGY `gemini-3.8-flash-medium`, Codex `gpt-6-luna` (medium); idea generators Sonnet and Flash; workers Haiku 5.5 or Flash; reviewer from a different family than the worker; Copilot excluded; Codex capped at about 15% of the budget.
- Stop rules: pause if any backend's tightest window is below 25%, or the study uses more than 30 points of the Claude 5-hour window. Creative deliverables count as full outputs, reported separately.

## 2. Why this was handed over (quota, measured 2026-11-07 05:51Z)
Claude weekly window 36% remaining (resets Nov 9 about 8:59pm Vancouver); Claude would cross the 25% floor within a few calls. Codex 5-hour 28% (resets about 07:04Z), weekly 45%. AGY Gemini 5-hour 85%, weekly 96%. Re-check with `usage_probe.py headroom` and `snapshot` before every phase.
**Open recommendation, NOT approved (decide with Richard):** rebalance clients so Claude is used for only two calls (Sonnet idea generation and Sonnet declaration): workers, successors and the formula scorer = Flash; reviewers and calibration raters = Luna after Codex's reset; run the pipeline unattended with few check-ins. If you adopt any change, make it BEFORE the freeze and record it in protocol.md as a disclosed change.

## 3. What exists (all in this folder)
`thresholds.json` (all frozen numbers) - `prompts.py` (every prompt and the rubric text) - `common.py` (model calls through the cli-agent router, transcript-based usage measurement) - `pool.py` (stripped candidate cards + novelty containment) - `scenarios.py` (claim A, seeded) - `choose.py` (selection; renamed from select.py because it shadowed the stdlib `select` module) - `analyze.py` (decision table and verdict) - `run_study.py` (driver: snapshot, calibrate, ideas, score, declare, item, status) - `pool-base.json` (30 cards: 25 PROPOSED non-Human-Gate missions + 5 inbox ideas), `formula-base.json` - `scenario-results-dev.json` (dev seed only) - `stop-baseline.json` (quota baseline at study start).
Machine-local helpers that may vanish (recreate): `/private/tmp/study060/codex-medium.sh` (a pass-through that injects only `-c model_reasoning_effort="medium"`; the router cannot set Codex effort), the empty `git init`-ed `/private/tmp/study060/neutral` directory, and `/private/tmp/study060/snapshot` (made by `run_study.py snapshot`).
Dev-checked on throwaway data only: `pool.py`, `scenarios.py dev`, `choose.py`, `analyze.py`. **NOT yet tested against live transcripts:** the `measure_*` usage functions in `common.py` and `run_study.py` itself. Test them on the first real call and fix before freeze.

## 4. Order of work (do not reorder)
1. Bootstrap per AGENTS.md. `manage_coordination.py checkout --mission MISSION-060 --agent codex --create-worktree --no-usage` (this marks the manifest IN_PROGRESS in your worktree, as intended). Heartbeat the lease; it lasts 60 minutes.
2. Quota check; settle the client allocation (section 2).
3. **Calibrate** the rubric on the dev cases 056/057/058: `python3 run_study.py calibrate` (currently raters haiku + flash; change to flash + luna if rebalancing). Needs raw agreement at least 0.80 over 9 judgments, else one disclosed prompt revision, else UNASSESSED. Calibration uses development cases only.
4. **Freeze**: write `protocol.md` (see section 5) and `freeze.json` (sha256 of every file listed in section 3 plus protocol.md). Commit and merge it BEFORE any held-out step, as a separate step from running anything (MISSION-056 lesson). Narrow the manifest's `projected_impact` to `LEARNING/experiments/mission-060/` and its own manifest at the same time.
5. Claim A: `python3 scenarios.py heldout` once, only after the freeze. Never re-run it to improve a number.
6. Prospective phase, in order: `snapshot`, `ideas`, `score`, `declare` (declarers run in the EMPTY neutral directory so they cannot see priority rankings; `declarations/attestation.json` records this), then `python3 choose.py`, then `run_study.py item ITEM-01` ... `ITEM-10` one at a time (each: worker, successor, reviewer; reviewers never receive the cohort label), then `python3 analyze.py`.
7. Record the verdict with `manage_backlog.py record-result` (CORROBORATED / FALSIFIED only; UNASSESSED means no verdict is recorded and the mission stays open with the reason). Capture the natural-observation follow-up in the inbox (it cannot be manufactured; 14-day window; it does not affect the verdict).
8. Findings and limitations go to a knowledge entry and Chronicle; lessons, interventions and repairs through `eval_gate.py --store .agent/learning/events.jsonl`; final check-in, integrate, then `soul_coverage.py closure`; export PR; no HF sync unless authorized.

## 5. Content for protocol.md: operational meanings and disclosed changes (write these in)
1. **D1** is operationalized as: the FINDING is new and goes beyond restating the brief; the question may already be a backlog question (otherwise every existing mission fails by definition). The manifest's wording "a question new to the knowledge base" is superseded here.
2. **B3** (formula share keeps 90%): ranks 1-11 are the same items under F and P and are not executed, so B3 holds by construction; the real diversion cost is the F-only cohort's outputs, reported.
3. **Interest-led slots** go to the top 5 by interest among candidates the formula would not select (formula rank above 16). P-only = those 5; F-only = formula ranks 12-16.
4. **Discretionary resources** for this study = the study's 16 slots.
5. **Additional tokens** = (last request's context - first request's context) + output tokens, measured from each client's own transcript; session scaffold (first request) and accumulated input reported separately; monetary cost unknown.
6. **Workers** run in a read-only repo snapshot with no AGENTS.md, CLAUDE.md, GEMINI.md or `.agent/` (no bootstrap, no always-on rules); worker scripts are NOT re-executed by the driver (untrusted code); the repo status is compared before and after each item.
7. **Claim A** is a seeded simulation; its blocker-reduction leg depends on the assumed 70% readiness-check recall (sensitivity at 40% and 100% is reported); only depth retention and high-interest count are non-trivial. It says nothing about real readiness checks.
8. **Idea round:** Sonnet and Flash each propose 2 ideas (4 total); each declarer rates every candidate except its own proposals (those ratings are excluded). Inbox and idea items have no manifest scores; Haiku (or the chosen scorer) estimates formula inputs before declarations, blind to interest.
9. **Originality** ratings are capped at 2 when a candidate's text overlaps the knowledge base by containment 0.35 or more.
10. **Natural observation** (8 opportunities, 14 days) is reported separately and does not affect the verdict.
11. **Calibration** is 3 dev cases x 3 checks x 2 raters; raw agreement, not kappa (too few judgments).
12. **Stop-rule reading:** any backend's tightest window below 25% pauses the study; ask Richard.

## 6. Pitfalls found so far (do not repeat)
- `select.py` shadows the stdlib `select` module; the file is `choose.py`.
- zsh does not word-split unquoted variables (a bug bit this session three times); use arrays or functions.
- Booting an agent inside the repo prints priority rankings in the SessionStart briefing, which would break blinding. Declarers must run in the neutral directory.
- Do not edit `thresholds.json`, `prompts.py`, `choose.py`, `analyze.py` or the rubric after the freeze; secondary tables cannot rescue a failed primary claim; UNASSESSED is not FALSIFIED.
- Shipping any selector or standing protected share is a SEPARATE later engineering step gated on the verdict and Richard's approval; record it as an inbox candidate, do not execute it, and do not create new missions.
- Never use Copilot. Unknown usage stays unknown.

## 7. Definition of done
A recorded verdict (or an explicit UNASSESSED with the failed gate), results.json, per-item assessments, usage log, findings and limitations preserved in the repo and soul export, lease checked in, and the engineering deliverable (instruments) reported separately from the empirical result.

## 8. Dispatching other CLIs: use the `cli-agent` skill (added after Richard's note)
`cli-agent` (`~/.claude/skills/cli-agent`, also installed for Codex and AGY; see AGENTS.md "Multi-Agent Orchestration") is how any agent launches a fresh session on another CLI; `common.py` and `run_study.py` already call its `run_agent.py`. Read its `SKILL.md` and `references/{codex,agy,claude}-cli-dispatch.md` before dispatching. Two limits to plan around: a Codex session runs in `workspace-write` with network disabled, so it cannot itself spawn `claude`/`agy`/`codex` unless Richard authorizes a wider sandbox (permission-bypass launches need explicit authorization and external containment); and the router cannot set Codex reasoning effort (use the `--executable` wrapper). If you are Codex, you may need Richard to run the pipeline commands from a Claude or plain shell session, or to authorize the wider sandbox; do not work around the sandbox.

## 9. Local implementation update

Current continuation: development calibration passed 9/9 with Flash/Luna and no
rubric revision. The candidate protocol/freeze package is prepared (37 repository
files, 31 runtime dependencies); public integration must still be verified before
any held-out execution. See PROGRESS-HANDOFF.md for latest commit/PR status.
Current wrappers are tracked claude_isolated.sh, codex_medium.sh and
contained_agy.py, superseding the old /tmp Codex helper path above. Earlier
not-started/uncalibrated descriptions are historical handoff context.

Live step-by-step continuation record: `PROGRESS-HANDOFF.md`. Update it immediately
after each completed step, including each prospective item, and before continuing.
It records the worktree, branch, current delivery/lease state and next action for
an AGY takeover if needed.

Live verification subsequently completed: Flash/Luna/Sonnet metering and the
actual contained Flash terminal canary passed. Ten development validation calls
used 14,140 measured additional tokens, including failures/repairs. See the live
handoff for new wrappers, source-token audit, containment scope and next action.
No calibration, freeze or held-out phase has run.

Richard subsequently authorized AGY Flash delegation and implementation of the
patches. See `PATCH-NOTES.md` for the implemented Flash/Luna allocation, instrument
repairs, 15 deterministic checks, and remaining metering/containment limitations.
The earlier client configuration and unimplemented-defect descriptions above are
historical baseline context. No study phase or freeze has run. Review the notes
before continuing; retain the original phase order and quota rule. Preparation
drafts are proposals, not authoritative protocol or calibration evidence.
