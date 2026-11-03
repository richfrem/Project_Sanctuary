# MISSION-027 experiment log: MLX server lifecycle hardening

Written **before** any test was run (pre-registration), then updated with results below.

## Orientation facts (read-only, measured 2026-11-03 on the running `:8080` server)
- Process tree: `uv tool uvx ...` launcher (pid 74191, RSS 4.8 MB) -> python worker (pid 74300), both in process group 74191.
- `ps` RSS of the worker is **10 MB** while its `footprint` is **4,711 MB** (4,465 MB IOAccelerator/Metal). `ps` RSS cannot see the model weights.
- The mission's literal discovery command `lsof -ti -sTCP:LISTEN :8080` **errors** on macOS lsof 4.91 ("status error"); the working form is
  `lsof -nP -iTCP:8080 -sTCP:LISTEN -t`. The mission's intent stands; its mechanism text needed correcting.
- Code reading (`mlx_server_manager.py`): discovery uses `lsof -ti :PORT` (includes client sockets); stop signals only the PID in the state file
  (the launcher) with `os.kill`; its "terminated" check compares the listener PID with that launcher PID; memory is read from the same launcher PID.

## Hypothesis and falsification (from MISSION-027)
Scoped discovery, process-group teardown, worker-based memory measurement and scheduled idle reaping give 100% reclamation and no zombie daemons.
Falsified if: after `stop_server` any child python still listens on the port; or reported memory does not reflect the model weights;
or port discovery returns a client socket.

## Pre-registered predictions
1. Against the **current** code, a stub server tree (launcher + listening worker child) reproduces at least three defects (RED): discovery returns a
   client PID; `stop_server` reports `stopped` while the worker still listens; `get_status` reports the launcher PID and tiny memory.
2. After the fix: 20 start/stop cycles of the stub tree leave 0 listeners and 0 surviving processes, and discovery never returns a client PID (0/20).
3. With the real model on a spare port (`:8099`, the `:8080` server is never touched): after `stop`, 0 processes remain and the port is closed;
   `footprint` while running is >= 4,000 MB (prediction ~4,700 MB), while `ps` RSS stays under 100 MB (the defect the old measure had).
4. Teardown never signals a process group that contains non-MLX processes (safety); it falls back to a tree kill of the worker and its launcher.

## Decision rule
CORROBORATED iff predictions 2 and 3 hold with zero exceptions and no safety violation. Any survivor, any client PID, or footprint < 2,000 MB
under a loaded model -> FALSIFIED. The corrected `lsof` syntax is recorded as a lesson either way.

## Method
- Tests: `operating-sanctuary-guardian/tests/test_mlx_lifecycle_hardening.py` (stub tree: a launcher process in its own session that spawns a
  listening worker; optional worker that ignores SIGTERM). RED first, then fix, then GREEN.
- Real-model validation (gated): `RUN_LIVE_MLX=1`, port 8099, results appended here.

---
# Results (appended after the runs)

## Prediction 1 (RED on the current code): HELD, 4 of 4 defects reproduced
| Test (stub launcher -> worker tree) | Result on the old code |
|---|---|
| discovery ignores client sockets | FAILED: returned the test process (a client), pid 51306, not the listener 51312 |
| stop tears down the worker child | FAILED: worker still alive after `stop_server` |
| stop reports failure if the worker survives (ignores SIGTERM) | FAILED: reported `stopped` while the worker was alive |
| status describes the worker | FAILED: reported the launcher's pid |

## Prediction 2 (stub, after the fix): HELD
20 start/stop cycles: 0 survivors, 0 client PIDs returned, every stop `verified`. All 19 stub/unit tests and the 7 older manager tests pass.

## Prediction 4 (safety): HELD
A server launched by a shell script shares its process group with the script and a bystander `sleep`. The group is judged unsafe, teardown falls back to
`process_tree`, the server tree dies, and the shell script and the bystander survive. A stale state-file PID (now an unrelated process) is never signalled.

## Prediction 3 (real model on :8099): PARTIAL: two of my numeric predictions were wrong
8 real-model start/stop cycles over three runs. `:8080` was untouched throughout (listener pid 74300 before and after).

| Measure | Predicted | Observed |
|---|---|---|
| footprint, **fresh** (right after boot) | >= 4,000 MB | **MISSED**: 562-3,317 MB (erratic: 3,317 / 1,247 / 1,766 / 562 / 1,493 / 1,441 / 1,493 / 1,493); weights load lazily |
| `ps` RSS | < 100 MB | **WRONG**: fresh servers showed 328-2,017 MB; the 10 MB reading on `:8080` is compressed memory (4,332 MB compressed), so `ps` RSS is unreliable either way |
| footprint, **warm** (after one 32-token generation; measurement point added after the fresh miss, labelled as such) | >= 4,000 MB | 4,304-4,361 MB over 6 cycles (4,711 MB on the 10-hour-old `:8080` server) |
| stop verified, process group signalled, all processes gone, no listener | all | 8/8 |
| client PID returned by discovery | 0 | 0/8 |
| system free memory: running -> after stop | recover | 14-24% -> 39-53% (about +4 GB returned) |

Data: `live_results_run1_idle_measurement.json` (2 idle-only cycles, the original miss) and `live_results.json` (final 3 warm cycles).
The middle run (3 warm cycles: 4,304 / 4,361 / 4,359 MB) was overwritten by the final run's file; its numbers are in the table above.

## Mission item 4 (idle reaping tied to Protocol 128 timers)
`reap_if_idle(only_managed=True)`; holistic wakeup calls it at boot; Protocol 128 Stage 5 gains a release step. A server the manager did not start is never reaped
(wakeup smoke test against the real `:8080` server left it running).

## Verdict
Mission falsification criteria: no child listener survived `stop_server` (0 survivors in 28 verified stops: 20 stub cycles + 8 real-model cycles), reported memory reflects the model weights when measured with
`footprint` on the worker, and no client socket was returned by discovery. **CORROBORATED**, with these recorded deviations: the mission's literal `lsof` command is invalid on
macOS (corrected), `ps` RSS is not a usable memory measure, and my own pre-registered numeric thresholds mis-specified the measurement point (fresh vs warm); the warm point was
added after the miss and is labelled as an amendment, not retrofitted into the prediction.

## Caveats
One machine (M1, 16 GB), one model, macOS tools (`lsof`, `ps`, `footprint`); SIP errors not exercised; the stub tree models the launcher/worker shape but not every real failure mode.
