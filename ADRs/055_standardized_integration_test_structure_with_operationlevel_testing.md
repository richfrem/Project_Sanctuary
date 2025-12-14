# Standardized Integration Test Structure with Operation-Level Testing

**Status:** proposed
**Date:** 2025-12-14
**Author:** Claude + User


---

## Context

Integration tests for MCP servers with LLM dependencies (agent_persona, council, orchestrator, forge_llm) are slow to execute. Running the full test suite every time is impractical during development. Developers need the ability to test individual MCP operations in isolation and track test execution history over time.

## Decision

We will standardize integration test structure across all 12 MCP servers with the following components:

1. **test_operations.py** - Tests each MCP operation individually with clear calling examples
2. **conftest.py** - Includes automatic test result logging to test_history.json
3. **test_history.json** - Auto-generated file tracking last 20 test runs with timestamps and results

Key features:
- Operations marked with @pytest.mark.slow for LLM-calling tests
- Run fast tests with: `pytest -m "not slow"`
- Run individual operation with: `pytest ::test_<operation> -v`
- Comprehensive docstrings with calling examples, operations table, and requirements
- SLOW_SERVERS list in test harness (agent_persona, council, orchestrator) skipped by default

Test harness (tests/run_all_tests.py) updated with --slow flag to include/exclude slow tests.

## Consequences

Positive:
- Faster development iteration by testing individual operations
- Clear visibility into which tests call LLMs vs pure logic
- Historical tracking of test results with timestamps
- Standardized structure across all MCP servers

Negative:
- Additional boilerplate files per server
- test_history.json files need to be in .gitignore (local-only)

Risks:
- Developers may forget to run full test suite before commits (mitigated by P101 pre-commit hook)
