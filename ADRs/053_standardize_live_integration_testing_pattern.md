# Standardize Live Integration Testing Pattern

**Status:** proposed
**Date:** 2025-12-14
**Author:** AI Assistant


---

## Context

ADR 048 mandated Live Integration Testing (Layer 2). We need a standardized way to ensure required services (ChromaDB, Ollama, Git) are running before executing these tests to prevent flaky failures and improve developer experience.

## Decision

We will adhere to a standard "Live Integration Test" architecture:
1. All Layer 2 integration tests MUST inherit from `LiveIntegrationTest` (defined in `tests/integration/live_test_base.py`).
2. This base class implements a `_check_deps` autouse fixture that verifies TCP connectivity.
3. Tests skip if dependencies are missing (unless CI=true).

## Consequences

Positive:
- Consistent test structure across all MCPs
- Automatic dependency verification
- Reduces boilerplate in individual test files

Negative:
- Requires refactoring existing integration tests
