# Spec 0004: Standardize CLI & Refactor RAG Ingest

## 1. Goal
Ensure `tools/cli.py` achieves 100% parity with `scripts/cortex_cli.py`, allowing the legacy script to be deprecated. Maintain business logic within `mcp_servers/` but expose all functionality via the unified CLI.

## 2. Problem
`scripts/cortex_cli.py` is the current "source of truth" for many operations, but `tools/cli.py` is the intended future entry point. Using two different CLIs causes confusion and maintenance drift. `tools/cli.py` is missing a few commands (`bootstrap-debrief`) and arguments (`--output` for debrief) present in the legacy script.

## 3. Solution (Revised)
1.  **Audit**: Perform gap analysis (Complete).
2.  **Fill Gaps**: Add missing commands and arguments to `tools/cli.py`.
    -   `debrief --output`
    -   `bootstrap-debrief` (wrapper around snapshot)
    -   `guardian wakeup --manifest`
3.  **Standardize Imports**: Ensure `tools/cli.py` correctly imports operations from `mcp_servers`.
4.  **Preserve Location**: Do **NOT** move `mcp_servers` code. Keep it where it is to support potential standard MCP server usage.

## 4. Key Changes
-   **Modify**: `tools/cli.py`
    - Add `bootstrap-debrief` command.
    - Update `debrief` handler to support file output.
    - Update `guardian` handler to support custom manifest path.

## 5. Verification
-   Run `tools/cli.py debrief --output test_debrief.md`
-   Run `tools/cli.py bootstrap-debrief --manifest ...`
-   Verify `tools/cli.py guardian wakeup --manifest ...`
