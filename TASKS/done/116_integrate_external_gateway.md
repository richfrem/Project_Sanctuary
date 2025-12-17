# TASK: Implement Dynamic MCP Gateway with IBM ContextForge

**Status:** complete
**Priority:** High
**Lead:** AI Assistant
**Dependencies:** Task 115 (complete), ADR 057, Protocol 122 (pending)
**Related Documents:** ADR 056, ADR 057, Task 115, Protocol 122 (pending), docs/mcp_gateway/, IBM ContextForge (https://github.com/IBM/mcp-context-forge)

---

## 1. Objective

Implement the Dynamic MCP Gateway Architecture by forking and customizing IBM ContextForge. Deploy MVP with 3 servers in Week 1, evaluate feature gaps and performance, customize for Sanctuary-specific requirements (allowlist, Protocol 101/114 integration), migrate all 12 servers, and deploy to production with monitoring. Achieve 88% context reduction (8,400 → 1,000 tokens) and <30ms latency overhead.

## 2. Deliverables

1. Forked sanctuary-gateway repository from IBM ContextForge
2. MVP deployment with 3 servers (Week 1)
3. Week 1 evaluation report (feature gaps, performance)
4. Sanctuary allowlist plugin (sanctuary_gateway/plugins/sanctuary_allowlist.py)
5. Protocol 114 integration hooks
6. All 12 servers migrated to Gateway
7. Side-by-side Claude Desktop configuration
8. Production deployment (Podman/systemd)
9. OpenTelemetry monitoring dashboard

## 3. Acceptance Criteria

- Week 1 MVP deployed with 3 servers (rag_cortex, task, git_workflow)
- ContextForge evaluation gate passed (feature gaps <50%, latency <50ms)
- Side-by-side deployment working in Claude Desktop
- All 12 servers migrated to Gateway
- Sanctuary allowlist plugin implemented (Protocol 101)
- Protocol 114 (Guardian Wakeup) integration complete
- Performance validated (<30ms latency overhead)
- E2E testing complete with Claude Desktop
- Production monitoring setup (OpenTelemetry)
- Documentation updated with deployment guide

## Notes

Implementation of Dynamic MCP Gateway using IBM ContextForge. 4-week timeline with Week 1 evaluation gate. Side-by-side deployment ensures zero-risk migration. All existing MCP server code remains unchanged. Fallback to custom build (Task 115 doc 07) if ContextForge evaluation fails.
**Status Change (2025-12-15):** backlog → todo
Ready to begin implementation. Waiting for Protocol 122 creation and user approval to proceed with Week 1 (fork ContextForge, deploy MVP).
**Status Change (2025-12-15):** todo → in-progress
Protocol 122 created. Ready to begin Week 1: Fork IBM ContextForge, deploy MVP with 3 servers (rag_cortex, task, git_workflow), evaluate feature gaps and performance.

**Status Change (2025-12-17):** in-progress → complete
Phase 1 (External Gateway Integration) complete. Gateway decoupled to external Podman service (ADR 058). Black box tests passing. ADR 060 (Fleet of 7) defines container architecture. Ready for Task 119 (sanctuary-utils pilot).
