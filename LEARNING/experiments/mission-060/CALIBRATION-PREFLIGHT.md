# Development calibration preflight — 2026-11-08 UTC

COMPLETED: first live calibration passed 9/9 agreement (1.0 >= 0.80), no prompt
revision. Six calls used 22,421 measured additional tokens; source audit PASS.
All nine paired judgments were positive, so this proves agreement on these
development cases, not evaluator specificity. See calibration-results.json,
usage-log.jsonl and verification/usage-source-audit.json. Earlier waiting/start
states below are historical. No freeze or held-out study ran.

05:33 UTC UPDATE: shared fix publicly merged in PR #757 and globally installed;
all three harness copies match merged source, all 33 installed tests PASS, live
reset-less Claude usage parses correctly. Step 2 is COMPLETE. Current native
Codex minimum 31%, Claude 35%, AGY 49.857%, all above floor. Calibration now
starting under existing authorization; the earlier waiting status below is
historical. No thresholds or rubric prompts changed before this first live run.

Status: PREPARED, waiting on shared parser deployment; no rater call dispatched.

Richard instructed: complete step 2 and begin step 3. The generic control plane
is verified disabled and consistent in this clone. The current quota snapshot
still has Claude status=unparsed because reset-less session usage is not recognized
by the installed shared parser. The checked-in companion fix c93f4b7b has 33
passing tests, but it is not pushed/merged/installed. Permission requested for
those exact publication/integration/global-install actions; no approval inferred.
Global APM update dry run completed with no writes. Do not weaken the quota rule
or edit installed copies directly to proceed.

The fresh zero-model-call Codex account endpoint was read at approximately 05:19
UTC: 3% five-hour used and 67% weekly used (97% / 33% remaining), ordinary usage
allowed. Evidence: verification/codex-current-quota-20261108.json. This supersedes
the stale rollout quota from the 05:17 snapshot. AGY's tightest bucket is 49.857%
remaining. Claude requires correct raw parsing; absence of a reset suffix is not
an unknown displayed percentage or permission to impute zero.

## Planned development run, before any held-out work

- Command from main shell with working CLI PATH:
  `python3 /Users/richardfremmerlid/Projects/Project_Sanctuary/.worktrees/codex-mission-060/LEARNING/experiments/mission-060/run_study.py calibrate`
- Only development artifacts from mission-056, mission-057, mission-058.
- Raters: Flash and Luna, separate model families; fresh neutral sessions.
- Nine paired D1/D2/D3 boolean judgments, six bounded CLI calls total.
- Pre-execution prediction: raw agreement at least 0.80. Falsification: lower
  agreement, missing/nonboolean judgments, unknown usage, failed quota/blinding
  checks or failed dispatch. These are gates, not scientific verdicts.
- At most one disclosed prompt revision if the reliability bound fails; if it
  still fails, report UNASSESSED and preserve the record. Do not retry a dispatched
  call ID or tune on held-out results.
- Existing usage-log.jsonl has 14,140 measured validation tokens. Haiku repair
  dispatch and orchestration are separate unknown costs, not zero.
- Record calibration-results.json and actual usage, then immediately update
  PROGRESS-HANDOFF.md. Freeze/integration and prospective execution are later steps.
