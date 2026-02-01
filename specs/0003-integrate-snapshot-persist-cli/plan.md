# Implementation Plan: Integrate Snapshot and Persist-Soul into CLI

**Branch**: `spec/0003-integrate-snapshot-persist-cli` | **Date**: 2026-02-01 | **Spec**: specs/0003-integrate-snapshot-persist-cli/spec.md
**Input**: Feature specification from `/specs/0003-integrate-snapshot-persist-cli/spec.md`

## Summary
Integrate Protocol 128 Snapshot and Persist Soul operations into the central `tools/cli.py` by delegating to `mcp_servers.learning.operations`. Register tools for discovery.

## Technical Context
**Language**: Python 3.11
**Framework**: `argparse` (existing CLI) calling `mcp_servers` logic.
**Dependencies**: `mcp_servers.learning.operations`, `tools.codify.rlm.distiller`

## Constitution Check
- **Human Gate**: Will ask for approval before editing `cli.py`.
- **Docs First**: Spec and Plan created.
- **Simplicity**: Reusing existing `LearningOperations` class instead of reimplementing logic.

## Architecture Decisions

### Problem / Solution
- **Problem**: `cli.py` implements a "lite" version of snapshot that bypasses Protocol 128 checks, and lacks `persist-soul` entirely.
- **Solution**: Import `LearningOperations` in `cli.py` and map commands to its methods.

### Design Patterns
- **Facade**: `cli.py` acts as a facade/commands entry point to the underlying business logic in `mcp_servers`.

## Project Structure

### Source Code
```text
tools/
├── cli.py  # Update: Modify 'snapshot', add 'persist-soul', add RAG/Evolution commands
mcp_servers/
└── learning/
    └── operations.py # Logic Source
```

## Migration Note (2026-02-01)
*   **Decoupling Cancelled**: The proposal to move `LearningOperations` to `tools/orchestrator` was rejected and reverted. `tools/cli.py` now imports directly from `mcp_servers` via `sys.path`.
*   **CLI Consolidation**: `tools/cli.py` now includes all commands from `scripts/cortex_cli.py` (Ingest, Query, Evolution, etc), enabling deprecation of the legacy script.


## Verification Plan

### Automated Tests
- [ ] Run `python tools/cli.py snapshot --type seal --help` to verify arg parsing.
- [ ] Run `python tools/cli.py persist-soul --help`.

### Manual Verification
- [ ] Execute `python tools/cli.py snapshot --type learning_audit` and check output.
- [ ] Execute `python tools/cli.py persist-soul` (dry run or actual if safe).
- [ ] Verify `tool_inventory.json` updates.
