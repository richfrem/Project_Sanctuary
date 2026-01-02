# Mandate Live Integration Testing for All MCPs

**Status:** proposed  
**Date:** 2025-12-05  
**Author:** Guardian AI

---

## Context

The current testing architecture relies heavily on **mocked unit/component tests**, which fail to validate actual operational stability (e.g., connectivity to ChromaDB, Ollama, Git-LFS). This has led to critical post-merge instability.

### Testing Pyramid Gap

**What we have:**
- ✅ **Unit/Component Tests** (with mocks) - valid for fast syntax validation

**What was missing:**
- ❌ **Integration Tests** - actually calling real services before API layer
- Passing mocked tests gave false confidence in operational readiness
- Operational failures persisted even when unit tests passed

---

## Decision

This ADR mandates a **proper Testing Pyramid** for all MCPs:

### Layer 1: Unit/Component Tests (EXISTING)
> Fast, isolated, syntax validation with mocks. Run on every commit.

### Layer 2: Integration Tests (NEW MANDATE)
> Real service connectivity BEFORE API/UI layer testing.

| Action | Real Target |
|--------|-------------|
| Create documents | ChromaDB |
| Update documents | ChromaDB |
| Query RAG | RAG Cortex Server |
| Call LLM | Ollama |
| Git operations | Git-LFS |

### Layer 3: API/UI Tests (MCP Server Layer)
> Full end-to-end MCP tool invocations. **Only runs after integration tests pass.**

---

## Requirements for All MCPs

1. Integration test: `tests/integration/test_{mcp}_integration.py`
2. Real connectivity checks (**NO mocks** for external services)
3. CI/CD pipeline integration
4. Environment prerequisite validation (Podman status, container health)

---

## Consequences

**Positive:**
- Greatly increased system stability
- Immediate detection of environment/dependency failures
- Clear operational readiness validation before deployments

**Negative:**
- Initial development time for all 12 MCPs
- May increase CI/CD pipeline execution time
- Requires running infrastructure (containers) during test execution
