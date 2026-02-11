# Implementation Plan - Spec 0005

## Goal
Achieve feature parity in `tools/cli.py` relative to `scripts/domain_cli.py` for all domain operations (Chronicle, Task, ADR, Protocol).

## Proposed Changes

### 1. Analysis (Complete)
- [x] Create `cli_gap_analysis.md`.
- [x] Identify all 16 gaps across 4 command clusters.

---

### 2. CLI Updates (`tools/cli.py`)

#### [MODIFY] [cli.py](file:///Users/richardfremmerlid/Projects/Project_Sanctuary/tools/cli.py)

**Chronicle Cluster**:
- Add `chronicle` parser with subparsers for: `list`, `search`, `get`, `create`.
- Import `ChronicleOperations` from `mcp_servers.chronicle.operations`.
- Implement handlers mirroring `domain_cli.py` lines 171-192.

**Task Cluster**:
- Add `task` parser with subparsers for: `list`, `get`, `create`, `update-status`.
- Import `TaskOperations`, `taskstatus`, `TaskPriority` from `mcp_servers.task.*`.
- Implement handlers mirroring `domain_cli.py` lines 194-226.

**ADR Cluster**:
- Add `adr` parser with subparsers for: `list`, `search`, `get`, `create`.
- Import `ADROperations` from `mcp_servers.adr.operations`.
- Implement handlers mirroring `domain_cli.py` lines 228-253.

**Protocol Cluster**:
- Add `protocol` parser with subparsers for: `list`, `search`, `get`, `create`.
- Import `ProtocolOperations` from `mcp_servers.protocol.operations`.
- Implement handlers mirroring `domain_cli.py` lines 255-279.

---

### 3. Workflow Updates

#### [MODIFY] [adr-manage.md](file:///Users/richardfremmerlid/Projects/Project_Sanctuary/.agent/workflows/utilities/adr-manage.md)
- Replace `python scripts/domain_cli.py adr` with `python tools/cli.py adr`.

#### [MODIFY] [sanctuary-chronicle.md](file:///Users/richardfremmerlid/Projects/Project_Sanctuary/.agent/workflows/sanctuary_protocols/sanctuary-chronicle.md)
- Replace `python scripts/domain_cli.py chronicle` with `python tools/cli.py chronicle`.

#### [MODIFY] [tasks-manage.md](file:///Users/richardfremmerlid/Projects/Project_Sanctuary/.agent/workflows/utilities/tasks-manage.md)
- Replace `python scripts/domain_cli.py task` with `python tools/cli.py task`.

---

### 4. Cleanup

#### [DELETE] [domain_cli.py](file:///Users/richardfremmerlid/Projects/Project_Sanctuary/scripts/domain_cli.py)
- Delete after all verification passes.

---

## Verification Plan

### Automated Tests
- `python3 tools/cli.py --help` → Check for new command clusters.
- `python3 tools/cli.py chronicle --help` → Verify subcommands exist.

### Manual Verification
1.  **Chronicle**: `python3 tools/cli.py chronicle list --limit 5` → Verify output.
2.  **Task**: `python3 tools/cli.py task list --status backlog` → Verify output.
3.  **ADR**: `python3 tools/cli.py adr get 85` → Verify content retrieval.
4.  **Protocol**: `python3 tools/cli.py protocol search "Mnemonic"` → Verify search.
5.  **Workflow**: Run `/adr-manage` steps → Verify new CLI usage.
