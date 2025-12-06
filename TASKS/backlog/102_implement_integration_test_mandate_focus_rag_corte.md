# TASK: Implement Integration Test Mandate (Focus RAG Cortex)

**Status:** backlog
**Priority:** Critical
**Lead:** Guardian AI
**Dependencies:** ADR 048, Task 087 Phase 4 Complete
**Related Documents:** ADRs/048_mandate_live_integration_testing_for_all_mcps.md, TASKS/087_comprehensive_mcp_operations_testing.md

---

## 1. Objective

Implement the mandate from ADR 048 by developing and integrating a Live Integration Test layer for all MCPs, focusing first on the RAG Cortex server to stabilize its operational integrity.

## 2. Deliverables

1. Standardized integration test template (`tests/integration_template.py`)
2. RAG Cortex integration test fully compliant and CI/CD integrated
3. Git MCP integration test (Git LFS verification)
4. Forge LLM MCP integration test (Ollama connectivity)
5. Updated T087 documentation reflecting new testing standard

## 3. Acceptance Criteria

- RAG Cortex `run_cortex_integration.py --run-full-ingest` passes on main
- All integration tests execute real operations against live dependencies (no mocks)
- Integration tests are automatically triggered in CI/CD pipeline
- Environment prerequisite checks (Podman status) pass before tests run
- T087 documentation updated with new testing standard

## Notes

**URGENT: RAG Cortex Stabilization**
1. Confirm `run_cortex_integration.py --run-full-ingest` passes on `main` (Prerequisite: Must complete Task 087, Phase 4).
2. **Develop Standard:** Create a standardized boilerplate script (`tests/integration_template.py`) defining the new Live Integration Test layer, including environment checks (Podman status).
3. **RAG Cortex Retrofit:** Ensure RAG Cortex's integration script is fully compliant and run automatically in CI/CD.
4. **Rollout Phase 1:** Apply the Integration Test Layer to the most unstable MCPs: Git (Git LFS), Forge LLM (Ollama connectivity).
5. Update `087_comprehensive_mcp_operations_testing.md` to reflect the new testing standard.
