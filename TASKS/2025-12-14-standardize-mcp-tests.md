Title: Standardize MCP Server Tests — Base Classes & Structure
Date: 2025-12-14
Owner: richardfremmerlid
Branch: feature/task-101-integration-test-base-refactor

Summary
-------
Establish and implement a standard testing structure for all MCP servers under `tests/mcp_servers/<server>/{unit,integration}` and document the use of two new base classes:

- `tests/mcp_servers/base/base_unit_test.py` (`BaseUnitTest`)
- `tests/mcp_servers/base/base_integration_test.py` (`BaseIntegrationTest`)

Objective
---------
1. Make the intended developer contract for the base classes explicit and discoverable.
2. Ensure every MCP server test suite follows the same directory layout and uses the base classes where appropriate.
3. Update documentation (`tests/README.md`) and relevant ADRs (`ADRs/053*`, `ADRs/054*`) to reflect the new standard.
4. Provide canonical example tests and, where feasible, refactor one canonical server (small, low-risk) to demonstrate the pattern.

Deliverables
------------
- New task file (this document).
- Updated `tests/README.md` with a "Base Test Classes" section and usage examples.
- ADR updates to reflect actual class names and file locations for the integration test base class.
- Canonical example unit & integration test templates committed under one server (e.g., `chronicle`).
- A small, minimal refactor of one server's tests to adopt the standard structure and inheritance (proof-of-concept).

Acceptance criteria
-------------------
- README contains a clear description and example for `BaseIntegrationTest` and `BaseUnitTest`.
- ADR 053 updated to point to `tests/mcp_servers/base/base_integration_test.py` and documents the `_check_deps` autouse fixture behavior (skip locally, fail in CI when `CI=true`).
- At least one server's tests (unit + integration) are shown as an example following the new layout and inheritance.
- No behavioral changes to unit tests; integration tests should still skip locally if required services are missing.

Plan / Steps
------------
1. Audit: list existing `tests/mcp_servers/*` directories and identify servers that deviate from the standard layout.
2. Documentation: update `tests/README.md` with base-class section and add usage examples.
3. ADRs: update ADR 053 and optionally ADR 054 to reference the real files and expected behavior.
4. Templates: add `test_example_unit.py` and `test_example_integration.py` templates for reuse.
5. POC Refactor: pick a low-risk server (proposed: `chronicle`) and refactor its tests to the new structure and inheritance; run unit tests.
6. Extend: propose a schedule for migrating the remaining servers (per PRs or sprints).

Risks & Notes
-------------
- Some servers may have bespoke fixtures or tight coupling that require minor import/path fixes. These will be handled per-server in targeted PRs.
- This task does not force-migrate all servers at once; it establishes the standard and a pattern for incremental refactors.

Related files
-------------
- `tests/mcp_servers/base/base_integration_test.py`
- `tests/mcp_servers/base/base_unit_test.py`
- `tests/README.md`
- `ADRs/053_standardize_live_integration_testing_pattern.md`
- `ADRs/054_harmonize_rag_cortex_test_structure.md`

Next actions (immediate)
------------------------
1. Confirm you want me to proceed with: README + ADR 053 edits, plus creating canonical example templates and a POC refactor of `chronicle`.
2. If confirmed, I will implement the README and ADR edits, add templates, and perform the POC refactor on your current branch.
