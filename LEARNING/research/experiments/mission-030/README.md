# MISSION-030 experiment: deterministic PreCompact snapshot hook

Pre-registered 2026-11-05 by claude-code, before any code was written.

## Predictions (each can fail)
- P1. The handler, fed a PreCompact payload that points at a real Claude Code transcript (this session's, several MB), finishes in under 2,500 ms in all 30 runs; median under 300 ms.
- P2. The snapshot lists every uncommitted file (git status) and contains the last assistant conclusions and at least one friction line when the transcript has them.
- P3. A planted token (hf_/sk-/ghp_/Bearer) never appears in the snapshot.
- P4. The handler exits 0 for malformed stdin, a missing transcript, an empty payload and a non-Sanctuary working directory (it must never block compaction).
- P5. `hooks.json` registers `PreCompact` and still validates for `SessionStart`.
- P6. Delegation: Codex (isolated dispatch, given only the handler source and the spec) writes a test file that passes against the handler on the first run. Prior: 50%.
- P7 (not testable in this session). The registered hook fires on a real compaction. Recorded as UNVERIFIED until a real `/compact` happens in an installed session; the hypothesis's first falsifier cannot be closed before that.

## Results
| Prediction | Result |
|---|---|
| P1 < 2,500 ms in 30/30 runs, median < 300 ms on a real multi-MB transcript | **Held**: 20.6 MB transcript, 30 runs, median 61 ms, p95 65 ms, max 67 ms |
| P2 snapshot has every uncommitted file, last conclusions, friction | **Held** (real run: branch, 5 uncommitted files, 5 conclusions, friction lines from failing tests) |
| P3 planted token never appears | **Held** (unit test by Codex and integration test; `[REDACTED]` in output) |
| P4 exit 0 for malformed stdin, missing transcript, empty payload, non-Sanctuary cwd | **Held** (no stdout, no `.agent/` created outside a Sanctuary checkout) |
| P5 `hooks.json` registers PreCompact and keeps SessionStart | **Held**; MISSION-024's 5 tests still pass |
| P6 Codex-written tests pass first run | **Held with caveats**: 11/11 passed on the first run, but only after I trimmed log lines my own extraction had copied into the file, and I changed one line (`HOOK_DIR` env var -> default to the hook directory). The tests themselves are Codex's. |
| P7 hook fires on a real compaction | **Held for `manual`**: a scratch Claude Code session ran `/compact` with the hook registered through a temporary `--settings` file; the handler wrote a snapshot (`trigger: manual`, 18 ms). The `auto` trigger (context-full compaction) was not exercised and is still UNVERIFIED. |

Also built: `session_start_wakeup.py` adds a "Pre-compaction snapshot" pointer when SessionStart `source == "compact"` (tested both ways).

Limits: the first falsifier ("a compaction occurs without invoking the hook") is closed only for a manual compaction on this CLI version. Installation matters: the hook runs only in sessions where the sanctuary-core plugin (hooks.json) is installed. The snapshot is provisional and unattended: conclusions are the last assistant text, not a curated summary.

## Measurement note (MISSION-035)
The Codex dispatch (isolated, `gpt-6-luna`, 28 s) is the first measured Codex record: 27,943 tokens (26,893 in / 1,050 out), $0.0032, from rollout files. Writing it exposed two defects, fixed here: `codex_tokens_between` returned only a total, so the hook priced output at the input rate (cost was $0.0028, now $0.0032); and my first `finish` call used the wrong lease key so it measured an empty window. The recorded `wall_seconds` is the lease window (126 s), not Codex's runtime (28 s). Quota windows moved codex 5-hour +11% and weekly +2% for those 28k tokens, which is one data point (not concurrent) and not a calibration.
