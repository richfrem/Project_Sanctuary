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
└── cli.py  # Update: Modify 'snapshot' and add 'persist-soul'
mcp_servers/
└── learning/
    └── operations.py # Logic source (no changes expected, just usage)
```

## Verification Plan

### Automated Tests
- [ ] Run `python tools/cli.py snapshot --type seal --help` to verify arg parsing.
- [ ] Run `python tools/cli.py persist-soul --help`.

### Manual Verification
- [ ] Execute `python tools/cli.py snapshot --type learning_audit` and check output.
- [ ] Execute `python tools/cli.py persist-soul` (dry run or actual if safe).
- [ ] Verify `tool_inventory.json` updates.
