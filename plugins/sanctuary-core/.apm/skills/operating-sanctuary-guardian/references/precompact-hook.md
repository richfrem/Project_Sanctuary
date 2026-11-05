# PreCompact snapshot hook (MISSION-030)

`plugins/sanctuary-core/.apm/hooks/pre_compact_snapshot.py`, registered under `PreCompact` in `hooks.json`.

Before the harness compacts the context it writes `.agent/learning/precompact/<UTC>-<session>.md` (gitignored, last 20 kept)
with: branch, uncommitted files, the last assistant conclusions and recent friction lines from the transcript tail, secrets
redacted. It is deterministic (no model call), prints nothing, always exits 0 so it can never block compaction, and is
silent outside a Sanctuary checkout (no `context/` directory).

After a compaction, `session_start_wakeup.py` (SessionStart `source == "compact"`) adds a pointer to the newest snapshot to
the injected context. The snapshot is PROVISIONAL: verify it against the repo.

Measured: median 61 ms, max 67 ms on a 20.6 MB transcript (ceiling 2,500 ms). Fired on a real manual `/compact`; the `auto`
trigger is unverified. It only runs in sessions where the sanctuary-core plugin hooks are installed.
Evidence: `LEARNING/experiments/mission-030/README.md`. Tests: `plugins/sanctuary-core/.apm/hooks/tests/`.
