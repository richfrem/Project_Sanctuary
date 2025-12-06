# Mandate Live Integration Testing for All MCPs

**Status:** proposed
**Date:** 2025-12-05
**Author:** AI Assistant


---

## Context

Passing unit tests (mocked dependencies) gave false confidence in operational readiness.
Operational failures persisted (e.g., database connection issues, I/O errors) even when tests passed.
The project lacks a dedicated, mandatory layer for integration testing.

## Decision

The current testing architecture relies heavily on mocked unit tests, which fail to validate actual operational stability (e.g., connectivity to ChromaDB, Ollama, and Git-LFS). This has led to critical post-merge instability and significant wasted time. This ADR mandates the implementation of a **Live Integration Test layer** for all MCPs to confirm real-world input/output against their primary external dependencies.

## Consequences

**Positive:** Greatly increased system stability; immediate detection of environment/dependency failures.
**Negative:** Requires initial development time to implement the integration layer for all 12 MCPs.
