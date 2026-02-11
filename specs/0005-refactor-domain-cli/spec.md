# Spec 0005: Migrate Domain CLI to tools/cli.py

## 1. Goal
Consolidate all domain CLI operations (Chronicle, Task, ADR, Protocol) into `tools/cli.py`, achieving 100% parity with `scripts/domain_cli.py` and allowing the legacy script to be deprecated.

## 2. Problem
`scripts/domain_cli.py` provides essential CRUD operations for domain entities (Chronicle, Task, ADR, Protocol), but it exists as a separate CLI entry point. This creates:
- Confusion about which CLI to use (`scripts/domain_cli.py` vs `tools/cli.py`)
- Maintenance drift between two CLI interfaces
- Inconsistent UX for users and agents

## 3. Solution
1.  **Audit**: Perform gap analysis (Complete - see `cli_gap_analysis.md`).
2.  **Migrate Commands**: Add all 16 missing commands to `tools/cli.py`:
    - `chronicle list|search|get|create`
    - `task list|get|create|update-status`
    - `adr list|search|get|create`
    - `protocol list|search|get|create`
3.  **Preserve Location**: Do **NOT** move `mcp_servers` code. Import operations directly.
4.  **Update Workflows**: Update `/workflow-*` files to use `tools/cli.py` instead of `domain_cli.py`.
5.  **Deprecate**: Delete `scripts/domain_cli.py` after verification.

## 4. Key Changes
-   **Modify**: `tools/cli.py`
    - Add `chronicle` command cluster with 4 subcommands.
    - Add `task` command cluster with 4 subcommands.
    - Add `adr` command cluster with 4 subcommands.
    - Add `protocol` command cluster with 4 subcommands.
-   **Modify**: `.agent/workflows/utilities/adr-manage.md`, `sanctuary-chronicle.md`, `tasks-manage.md`
    - Update command references from `domain_cli.py` to `tools/cli.py`.
-   **Delete**: `scripts/domain_cli.py` (after verification).

## 5. Verification
-   Run `tools/cli.py chronicle list --limit 5`
-   Run `tools/cli.py task list --status backlog`
-   Run `tools/cli.py adr get 85`
-   Run `tools/cli.py protocol search "Mnemonic"`
-   Verify all `/workflow-*` commands work with new CLI.
