# Orchestrator MCP Tests

Server-specific integration tests for orchestration behaviors (Strategic Crucible Loop, loop hardening, and policy validation).

Key integration tests:

- `integration/test_strategic_crucible_loop.py` — Full Strategic Crucible Loop simulation (gap analysis, ingestion, adaptation, synthesis).
- `integration/test_056_loop_hardening.py` — Orchestrator loop hardening scenarios (ingestion failure, commit integrity).

Run:

```bash
pytest tests/mcp_servers/orchestrator/integration/ -v -m integration
```

Notes:

- These tests are server-scoped integration tests and were moved from the top-level `tests/integration/` directory to keep multi-MCP scenarios separate from single-server integration tests.
