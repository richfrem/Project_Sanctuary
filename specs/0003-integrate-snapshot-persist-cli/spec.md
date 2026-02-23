# Feature Specification: Integrate Snapshot and Persist-Soul into CLI

**Feature Branch**: `spec/0003-integrate-snapshot-persist-cli`
**Category**: Feature
**Created**: 2026-02-01
**Status**: Draft
**Input**: User request to integrate snapshot and persist_soul from MCP operations into tools/cli.py

## User Scenarios & Testing

### User Story 1 - Snapshot Command Integration (Priority: P1)

As a developer, I want to use `python tools/cli.py snapshot --type seal` to create a learning snapshot that adheres to Protocol 128 (using `ops.capture_snapshot`), so that I don't have to rely on legacy scripts or raw bundle commands.

**Why this priority**: Core compliance requirement for closing sessions.

**Independent Test**: Run `tools/cli.py snapshot --type seal` and verify `learning_package_snapshot.md` is created and the operation validated by Protocol 128 logic.

**Acceptance Scenarios**:
1. **Given** a clean git state, **When** I run `cli.py snapshot --type seal`, **Then** it delegates to `LearningOperations.capture_snapshot` and produces a sealed snapshot.
2. **Given** a dirty git state (if protocol forbids), **When** I run the command, **Then** it should fail/warn as per protocol rules.
3. **Given** arguments, **When** I run `cli.py snapshot --type learning_audit`, **Then** it generates the audit packet.

---

### User Story 2 - Persist Soul Command Implementation (Priority: P1)

As a developer, I want to use `python tools/cli.py persist-soul` to broadcast my learning snapshots to Hugging Face, ensuring my session data is preserved.

**Why this priority**: Mandatory step for Protocol 128 phase VI.

**Independent Test**: Run `tools/cli.py persist-soul` and verify `data/soul_traces.jsonl` is updated and files are synced.

**Acceptance Scenarios**:
1. **Given** a sealed snapshot, **When** I run `cli.py persist-soul`, **Then** it calls `LearningOperations.persist_soul` to sync data.
2. **Given** no new data, **When** I run it, **Then** it should gracefully handle the state (idempotency).

---

### User Story 3 - Tool Registration (Priority: P2)

As a system, I want all relevant CLI tools and functions registered in `tool_inventory.json` and `rlm_tool_cache.json` using the distiller, so that the agent can discover and use them.

**Why this priority**: Required for agentic discovery ("Late-Binding" environment).

**Independent Test**: Check `plugins/tool-inventory/skills/tool-inventory/scripts/tool_inventory.json` for `cli.py` and related scripts.

**Acceptance Scenarios**:
1. **Given** `cli.py` and `distiller.py`, **When** I run the registration task, **Then** `tool_inventory.json` contains up-to-date entries for these tools.

## Requirements

### Functional Requirements
- **FR-001**: `cli.py` MUST import and use `mcp_servers.learning.operations.LearningOperations` for snapshot and persist logic.
- **FR-002**: `snapshot` command MUST support all Protocol 128 types (`seal`, `audit`, `learning_audit`, etc.).
- **FR-003**: `persist-soul` command MUST be added to `cli.py`.
- **FR-004**: Tools MUST be registered using `manage_tool_inventory.py` and `distiller.py`.

### Key Entities
- **LearningOperations**: The logic provider in `mcp_servers/learning/operations.py`.
- **CLI**: The entry point in `tools/cli.py`.

## Success Criteria
- **SC-001**: `cli.py snapshot` functionality matches the legacy `cortex_capture_snapshot` MCP tool behavior.
- **SC-002**: `cli.py persist-soul` functionality matches the legacy `cortex_persist_soul` MCP tool behavior.
