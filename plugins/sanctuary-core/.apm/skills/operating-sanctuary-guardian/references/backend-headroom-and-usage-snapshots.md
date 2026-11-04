# Backend Headroom and Usage Snapshots (pointer)

Single source of truth: `plugins/sanctuary-core/.apm/skills/evaluating-agent-work/references/usage-telemetry-sources.md`. This page only says where the hooks sit.

- **Checkout** (`manage_coordination.py checkout`): takes a zero-token usage snapshot (stored on the lease, cached for the briefing) and returns `usage_headroom_warnings` for any backend under 15% remaining, flagging the agent's own backend. Treat a warning on your own backend as a reason to hand the work to another CLI or wait for the reset.
- **Boot briefing**: the Predecessor Relay section shows a "Fuel gauge" line from the cached snapshot (no CLI calls at boot). If it says no reading under 6 hours old, run `usage_probe.py headroom`.
- **Checkin** (`manage_coordination.py checkin --task-class ... [--tokens N] [--quality 0-3] [--model M]`): second snapshot, window deltas, wall-clock, tokens, recorded as a measured entry.
- Disable with `--no-usage` or `SANCTUARY_NO_USAGE=1` (the test suite does); failures never block a lease operation.
