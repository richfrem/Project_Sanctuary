# Exact commands and observed outcomes

Commands ran from the codex-b-mission-050 worktree. No git mutation occurred.

## Historical commands (do not refreeze an existing run)

```sh
python3 /Users/richardfremmerlid/.codex/skills/running-control-plane/scripts/control_plane_mode.py status
```
Outcome: INCONSISTENT; database and git hooks missing. Exit 1. No repair attempted;
registration/ledger writes are outside worker scope. Boot was run under the same
exclusive Python `fcntl.flock('/tmp/sanctuary-mlx.lock')` used for inference:

```sh
python3 plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/scripts/guardian_wakeup.py --mode HOLISTIC
```
Outcome: GREEN; all 52 manifests valid, feature branch, local MLX ACTIVE. A locked
Python `urllib.request.urlopen('http://127.0.0.1:8080/v1/models', timeout=10)` returned
`mlx-community/Qwen2.5-Coder-7B-Instruct-4bit`. No server lifecycle action occurred.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 LEARNING/experiments/mission-050/v2/real_agent.py freeze
```
Outcome: FROZEN at 2026-11-06T05:02:29Z. Design, sources and fixtures hashed before
measurement; v1 file hashes captured. The freeze command intentionally refuses to
run twice. Do not recreate fixtures or change hashes to resume.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 LEARNING/experiments/mission-050/v2/run_simulation.py > LEARNING/experiments/mission-050/v2/simulation-run.log 2>&1
```
Outcome: exit 0; 1,620 trials; 5.477936082985252 measured wall seconds; raw evidence
SHA-256 `489e4c7727c05356c6d30c17e9202dbd16ac5f2a9d59d086f74ab2285c5ced8f`.
All simulation thresholds did not pass: gain, acknowledged loss and control miss.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -u LEARNING/experiments/mission-050/v2/real_agent.py run > LEARNING/experiments/mission-050/v2/real-run.log 2>&1
```
Outcome: see final `handoff.md`, `real-run.log`, `real-summary.json`, `audit.json`.
Each subgoal and checkpoint makes a fresh actual model request. No inference is
replaced by a cached experimental response. Every request uses exclusive flock.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 LEARNING/experiments/mission-050/v2/checkpoint_supervisor.py --stop-at-utc 2026-11-06T06:12:00+00:00 > LEARNING/experiments/mission-050/v2/checkpoint-supervisor.log 2>&1
```
Outcome: operational deadline guard only, not a changed experiment parameter.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider plugins/sanctuary-core/.apm/skills/operating-sanctuary-guardian/tests/test_adaptive_checkpoint.py LEARNING/experiments/mission-050/v2/test_real_agent.py --basetemp=LEARNING/experiments/mission-050/v2/pytest-final-tmp > LEARNING/experiments/mission-050/v2/validation-all.txt 2>&1
```
Outcome: **36 passed in 1.79s**. Includes exact v1 replay and v2 prior causality.

```sh
PYTHONDONTWRITEBYTECODE=1 python3 LEARNING/experiments/mission-050/v2/verify_v2.py > LEARNING/experiments/mission-050/v2/audit-final.txt 2>&1
PYTHONDONTWRITEBYTECODE=1 python3 LEARNING/experiments/mission-050/v2/report_results.py
PYTHONDONTWRITEBYTECODE=1 python3 LEARNING/experiments/mission-050/v2/finalize.py > LEARNING/experiments/mission-050/v2/finalize-output.json
```
Outcomes: final audit, threshold report, measured accounting, provisional ROI review
and SHA-256 seal are separate artifacts. `audit.json` states exact real trial/call
scope, incorrect answers, unique server response IDs, frozen/v1 integrity and the
1,620 exact simulation replays. See handoff for whether inference is unfinished.

## Resume only completed-trial checkpoints

After the prior client runner has exited and the explicitly marked operational
stop file has been removed, run the SAME frozen code, appending the log:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -u LEARNING/experiments/mission-050/v2/real_agent.py run >> LEARNING/experiments/mission-050/v2/real-run.log 2>&1
```

Completed trials are skipped; partial real snapshots are rejected deliberately.
Do not delete a true partial snapshot and rerun its trial. If a crash left an
unfinished trial, reconstruct its state from raw responses/events or report it.
Do not rerun freeze, change the source/design hashes, tune on held-out data, or
reuse these held-out measurements to select a new v2 candidate.

Re-run only audit/report/finalize after additional inference. Their outputs are
rebuildable projections of immutable raw calls and append-only completed trials.
The old seal becomes superseded; preserve it beside the new seal when resuming.
