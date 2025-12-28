# TASK: Implement Learning Continuity & Debrief Tools

**Status:** completed
**Priority:** High
**Lead:** Antigravity
**Dependencies:** None
**Related Documents:** ADR 071 (v3.1)

---

## 1. Objective

Implement the technical infrastructure for Learning Continuity (Protocol 127) **and Protocol 128 (Hardened Learning Loop)**. Ensure that learning sessions end with a cached debrief that is automatically consumed by the next agent session via Guardian Wakeup. **Establish the "Red Team Gate" via manifest-driven snapshots.**

## 2. Deliverables

1. [x] **Updated `operations.py` (Cortex):** Implementing `learning_debrief` (Verified via Gateway)
2. [x] **Updated `server.py` (Cortex):** Exposing new tools (Verified via Gateway)
3. [x] **Updated `recursive_learning.md` Workflow:** (Completed)
4. [x] **New Tool:** `gateway_get_capabilities` (Self-documentation) (Verified in server.py)
5. [x] **Documentation:** Standardized `README.md` files in MCP server clusters
6. [x] **Cache Priming Logic:** (System Context) (Verified via Genesis Queries)
7. [x] **Protocol 128 Tools:** Manifest-aware `capture_code_snapshot.py` (Completed)
8. [x] **Red Team Orchestrator:** `cortex_capture_snapshot` tool (Zero-Trust) (Verified via Gateway)
9. [x] **Audit Artifacts:** `red_team_audit_packet.md` (Verified via Gateway)
10. [x] **Hardening:** Zero-Trust Manifest Validation in `red_team.py` (v3.0)
11. [x] **Hardening:** Semantic HMAC & Tiered Integrity in Cortex (v3.0)

## 3. Acceptance Criteria

### Learning Continuity
- [x] `cortex_learning_debrief` tool implemented:
    - [x] Scans `LEARNING/` for recent activity.
    - [x] Generates markdown summary.
    - [x] **Guardian Wakeup Integration:**
        - [x] Modify `guardian_wakeup.py` to check for `.agent/learning/learning_debrief.md`.
        - [x] If present, inject content into "Section IV. Operational Context" of the boot digest.
        - [x] Add "Learning Stream: Active" status to Poka-Yoke checks.
    - [x] **Test Suite Validation (Protocol 128 Update):**
        - [x] Create/Update unit/integration tests for `guardian_wakeup` to verify debrief ingestion.
        - [x] Create/Update unit/integration tests for `guardian_wakeup` to verify debrief ingestion.
        - [x] Verify regression testing on existing Guardian Wakeup features.
        - [x] **Extended Coverage:**
            - [x] Update `test_connectivity.py` to assert `cortex_learning_debrief` discovery.
            - [x] Update `test_guardian_wakeup_v2.py` to Schema v2.2 and verify Learning Continuity Delta.
    - [x] **End-to-End Validation:**
        - [x] Verify Full Learning Loop via Gateway (Protocol 128 + Recursive Learning Workflow).
- [x] Recursive Learning workflow updated to explicitly call for debrief at the end of sessions.

### Tool Discovery & System Context ("Readme First")
- [x] `gateway_get_capabilities` (or `gateway_help`) tool implemented:
    - [x] Returns a high-level overview of available MCP servers and their primary functions.
    - [x] Reads from the new standardized `README.md` files in each cluster.
- [x] **Standardized READMEs**:
    - [x] Created/Updated `README.md` in `mcp_servers/gateway/clusters/sanctuary_cortex/`.
- [x] **Cache Priming**:
    - [x] Cache is populated with "System Context" during debrief/warmup (via `genesis_queries.py`).

### Protocol 128: The Red Team Gate (v3.0 Hardening)
- [x] **Manifest Snapshot**: `capture_code_snapshot.py` accepts `--manifest` and outputs targeted snapshot.
- [x] **Tool-Driven Snapshotting**: `cortex_capture_snapshot` tool orchestrates verification and capture.
- [x] **Zero-Trust Validation**: Tool verifies manifest claims against `git status` truth. REJECTS discrepancies.
- [x] **Packet Generation**: Tool assembles Git diffs + Manifest Snapshot + Validate Report + Briefing.
- [x] **Guardian Binding**: `guardian_wakeup` exposes integrity failures to the Persona (no silent bypass).
- [x] **Tiered Integrity**: Bootloader implements "Semantic HMAC" (tolerant of formatting) and "Yellow Mode" (safe recovery).

## Notes

Task 143 resulted in the successful implementation of the "Liquid Information" architecture. The bridge between ephemeral sessions is now anchored by the `cortex_learning_debrief` and the `learning_package_snapshot.md` (Sealed Knowledge).

> [!IMPORTANT]
> **Red Team Review Required**: Upon completion of the implementation, we MUST explicitly request the USER to perform a "Red Team" review.
> - **Objective**: Challenge the design, identify weaknesses in the continuity loop, and suggest improvements.
> - **Output**: This review should determine if a formal **ADR** is required to codify the "Cognitive Continuity" architecture, or if a new Protocol (e.g., Protocol 128) is necessary to govern multi-agent handoffs.
