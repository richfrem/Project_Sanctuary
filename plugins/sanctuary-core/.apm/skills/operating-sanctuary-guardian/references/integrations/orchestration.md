# Orchestration & Task Execution Integration

In Project Sanctuary, session orientation, governance, and final closure belong exclusively to the Guardian in the main conversation thread.

## Execution Topology
- Multi-agent coordination and model routing are delegated to the globally installed `orchestrating-agent-work` and `cli-agent` skills.
- The Guardian oversees the high-level Protocol 128 cycle, ensuring that work packages are verified before technical sealing.

## Shared Learning Loop (MISSION-055)
- The generic loop is companion-owned: `orchestrating-agent-work` (contract, lesson retrieval, redispatch, integration within authorization), `cli-agent` (enforced launch controls and structured results) and `evaluating-agent-outcomes` (findings, gates, cost, lessons).
- Guardian policy on top: take a flight-board lease and worktree first; use `usage_probe.py headroom` to choose a backend and set a token ceiling; the Human Gate is not widened by findings; engineering delivery is recorded with `sanctuary_adapter.py deliver` and never becomes an empirical verdict. Details: `evaluating-agent-work/references/shared-loop-adapter.md`.
- You are responsible for keeping these skills correct in both repositories (AGENTS.md, Protocol 133).
