# Adoption of IBM ContextForge for Dynamic MCP Gateway

**Status:** proposed
**Date:** 2025-12-15
**Author:** Project Sanctuary Core Team


---

## Context

Project Sanctuary faces critical scalability and context efficiency challenges with its current static MCP server loading approach. With 12 MCP servers exposing 63 tools, the context window overhead is ~8,400 tokens (21% of available context), limiting scalability to ~20 servers before context saturation.

**Problem Statement:**
- Context window saturation prevents scaling beyond 20 servers
- Static 1-to-1 binding (1 config entry = 1 server) is inflexible
- 180+ lines of manual JSON configuration is error-prone
- Security policies fragmented across 12 servers
- No centralized audit trail or monitoring

**Strategic Imperative:**
Protocol 125 (Autonomous Learning) requires rapid integration of new tools and data sources with minimal context overhead. The current architecture cannot support the planned expansion to 100+ tools.

**Research Conducted:**
Comprehensive research phase (Task 115) produced 12 documents (58,387 tokens) analyzing:
- MCP protocol and transport layers
- Production gateway implementations (Skywork.ai, Gravitee)
- Performance benchmarks and latency analysis
- Security architecture and threat modeling
- Current vs future state comparison
- Build vs buy vs reuse analysis

**Options Evaluated:**
1. Build custom gateway from scratch (6-8 weeks, $24K-32K)
2. Buy commercial solution - Operant AI (vendor lock-in, no budget)
3. Reuse IBM ContextForge (2-3 weeks, open-source)
4. Adapt general API gateway - Kong/APISIX (4-6 weeks, wrong stack)

**Decision Criteria:**
- Time to market (25% weight)
- Cost (20% weight)
- Customization flexibility (20% weight)
- Maintenance burden (15% weight)
- Security (10% weight)
- Vendor lock-in risk (10% weight)

## Decision

We will adopt IBM ContextForge (https://github.com/IBM/mcp-context-forge) as the foundation for Project Sanctuary's Dynamic MCP Gateway Architecture.

**Implementation Approach:**

### Week 1: Fork, Deploy, Evaluate (MVP)
```bash
# 1. Fork the repository
git clone https://github.com/IBM/mcp-context-forge.git sanctuary-gateway
cd sanctuary-gateway

# 2. Install dependencies
pip install -e .

# 3. Configure for Sanctuary
cp .env.example .env
# Edit .env with Sanctuary-specific settings

# 4. Deploy MVP with 3 servers
# Configure rag_cortex, task, git_workflow as backend servers

# 5. Test with Claude Desktop
# Update claude_desktop_config.json to use ContextForge gateway

# 6. EVALUATION GATE (End of Week 1)
# - Feature gaps <50%? ✅ Continue with ContextForge
# - Feature gaps >50%? ❌ Pivot to custom build
# - Performance <50ms? ✅ Continue
# - Performance >50ms? ❌ Optimize or pivot
```

### Week 2-3: Customize for Sanctuary
**Customization Areas:**
1. **Allowlist Plugin** - Tool-level security (Protocol 101 enforcement)
   ```python
   # sanctuary-gateway/plugins/sanctuary_allowlist.py
   class SanctuaryAllowlistPlugin:
       async def validate_tool_call(self, tool_name: str, args: dict) -> bool:
           """Validate tool call against project_mcp.json allowlist."""
           if tool_name not in self.allowlist.get("allowed_tools", []):
               raise PermissionError(f"Tool {tool_name} not in allowlist")
           
           # Check if operation requires approval (Protocol 101)
           if self.requires_approval(tool_name):
               return await self.request_approval(tool_name, args)
           
           return True
   ```

2. **Registry Integration** - SQLite or adapt ContextForge's built-in registry
3. **Protocol 114 Integration** - Guardian Wakeup hooks
4. **Sanctuary Metadata** - Chronicle/ADR/Protocol tracking
5. **Custom Routing Logic** - If needed for special cases

### Week 4: Production Hardening
- Migrate all 12 servers to Gateway
- Full integration testing (E2E with Claude Desktop)
- Performance optimization (<30ms latency target)
- Monitoring setup (OpenTelemetry)
- Documentation updates

### Side-by-Side Deployment Strategy
**Zero-Risk Migration:**
```json
{
  "mcpServers": {
    "sanctuary-broker": {
      "displayName": "🆕 Sanctuary Gateway",
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "sanctuary_gateway.server"]
    },
    "rag_cortex_legacy": {
      "displayName": "📦 RAG Cortex (Legacy)",
      "command": "/path/to/.venv/bin/python",
      "args": ["-m", "mcp_servers.rag_cortex.server"]
    }
    // ... other 11 legacy servers (unchanged)
  }
}
```

**Benefits:**
- Both old and new systems run simultaneously
- Easy A/B testing and comparison
- Instant rollback (<5 minutes)
- No changes to existing MCP server code

**Rationale:**
- **License:** Apache 2.0 (permissive, no vendor lock-in)
- **Adoption:** 3,000 stars, 434 forks, 88 contributors (strong community)
- **Activity:** 11 releases, latest v0.9.0 (Nov 2025) - very active
- **Leadership:** IBM-backed, enterprise-grade quality
- **Feature Overlap:** 90% match with Sanctuary requirements
- **Time Savings:** 2-3 weeks faster than building from scratch
- **Cost Savings:** $8,000-16,000 vs custom development
- **Production-Ready:** 400+ tests, security policy, migration guides
- **Technology Alignment:** Python/FastMCP matches our stack
- **Container-Native:** Supports Podman/Docker/Kubernetes/OpenShift

**Container Runtime Strategy:**
Architecture is container-runtime agnostic:
- Phase 1 (MVP): Podman Compose (local development)
- Phase 2 (Production): Podman with systemd (single host)
- Phase 3 (Scale): Kubernetes/OpenShift (multi-host, cloud)

**Decision Matrix Score:** 92% (highest among all options)

**Validation:**
Decision validated by Gemini 2.0 Flash Experimental (frontier model) and documented in formal decision document (docs/mcp_gateway/research/12_decision_document_gateway_adoption.md).

## Consequences

**Positive Consequences:**

1. **Context Efficiency (88% reduction)**
   - Current: 8,400 tokens → Future: 1,000 tokens
   - Frees 7,400 tokens for actual work
   - Enables scaling to 100+ servers

2. **Accelerated Time-to-Market**
   - 4 weeks total vs 6-8 weeks for custom build
   - 50% faster implementation
   - Earlier ROI realization

3. **Cost Savings**
   - $16,000 implementation vs $24,000-32,000 custom
   - $8,000-16,000 saved
   - 270% ROI in first year

4. **Production-Ready Foundation**
   - 400+ tests, battle-tested
   - IBM backing ensures quality
   - Active community support (88 contributors)

5. **Enterprise Features Included**
   - Multi-tenancy with RBAC
   - SSO integration (OAuth 2.0)
   - OpenTelemetry observability
   - Redis-backed federation
   - Admin UI for management

6. **Container Runtime Flexibility**
   - Works with Podman, Docker, Kubernetes, OpenShift
   - Easy migration path from local to cloud
   - No vendor lock-in

7. **Security Improvements**
   - Centralized security enforcement
   - Built-in auth, retries, rate-limiting
   - Comprehensive audit logging
   - Security policy documented

**Negative Consequences:**

1. **Learning Curve**
   - Large codebase to understand (~10K+ lines)
   - Need to learn ContextForge architecture
   - Mitigation: Comprehensive documentation, active community

2. **Customization Required (20% feature gap)**
   - Need to adapt RBAC to tool-level allowlist
   - Protocol 101/114 integration required
   - Mitigation: Well-documented plugin architecture

3. **Potential Over-Engineering**
   - ContextForge has features we don't need (multi-tenancy, SSO)
   - May add unnecessary complexity
   - Mitigation: Use only needed features, disable others

4. **Dependency on External Project**
   - Reliant on IBM's continued maintenance
   - Breaking changes in future versions
   - Mitigation: Apache 2.0 allows forking, active community ensures longevity

5. **Migration Effort**
   - All 12 servers need configuration updates
   - Testing required for each server
   - Mitigation: Side-by-side deployment, phased rollout

**Risk Mitigation:**

1. **Week 1 Evaluation Gate**
   - If feature gaps >50%, pivot to custom build
   - Performance must be <50ms latency
   - Security vulnerabilities must be addressable

2. **Fallback Plan**
   - Keep custom build plan (07_implementation_plan.md) ready
   - Can pivot if ContextForge doesn't meet needs
   - Apache 2.0 allows forking if IBM abandons project

3. **Side-by-Side Deployment**
   - Run Gateway alongside existing static config
   - Zero downtime migration
   - Easy rollback if issues arise

**Success Metrics:**
- ✅ Context overhead <1,500 tokens (from 8,400)
- ✅ All 12 servers migrated successfully
- ✅ Latency overhead <50ms (target: 15-30ms)
- ✅ Zero downtime during migration
- ✅ Protocol 101 enforcement functional
- ✅ Documentation complete

**Related Documentation:**
- Research: docs/mcp_gateway/research/ (12 documents)
- Decision Document: docs/mcp_gateway/research/12_decision_document_gateway_adoption.md
- Architecture: docs/mcp_gateway/architecture/ARCHITECTURE.md
- Task: TASKS/in-progress/115_design_and_specify_dynamic_mcp_gateway_architectur.md
