# Scratchpad

## User Instructions (Pending/Tracking)
- [ ] Review `mcp_servers/learning/operations.py`, `mcp_servers/learning/server.py`, `scripts/cortex_cli.py`, `mcp_servers/gateway/clusters/sanctuary_cortex/server.py` for "persist soul" and "hugging face" operations.
- [ ] Ensure `cli.py` includes operations accessible in `cortex_cli.py` to make them accessible to workflows.
- [ ] Specific operations to include:
    - `capture_snapshot`
    - `seal` (Note: verify if this needs a distinct command alias or if `snapshot --type seal` is sufficient)
    - `persist_soul`
    - `persist_soul_full`
- [ ] Policy Violation Note: Stopped autonomous git pushes. Changes must be verified first.

## Analysis Notes
- `cortex_cli.py` implements:
    - `snapshot` -> `LearningOperations.capture_snapshot` (with Iron Core Check)
    - `persist-soul` -> `LearningOperations.persist_soul` (with valence/uncertainty/full-sync)
    - `persist-soul-full` -> `LearningOperations.persist_soul_full`
    - `bootstrap-debrief` (Uses `capture_snapshot` with seal)
    - `ingest` / `query` / `stats` / `cache-stats` (Cortex RAG - Out of scope for this spec?)
    - `guardian` (Learning Loop)
    - `debrief` (Learning Loop)
    - `evolution` (Evolution MCP)

- Current `cli.py` implementation (Local Branch):
    - `snapshot`: Implemented (Matched `cortex_cli.py` logic, added Iron Core check).
    - `persist-soul`: Implemented (Added valence, uncertainty, full-sync args).
    - `persist-soul-full`: Implemented (Added command).
    - `seal`: Currently handled via `snapshot --type seal`. *Question: Does user want a top-level `seal` alias?*

## Questions for User
- Should `ingest` and `debrief` also be moved/copied to `cli.py` now, or is the scope strictly "persist/snapshot"?
- Is `snapshot --type seal` sufficient, or do you want a dedicated `cli.py seal` command?
