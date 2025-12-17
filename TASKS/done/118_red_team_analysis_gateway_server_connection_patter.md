# TASK: Red Team Analysis: Gateway Server Connection Patterns

**Status:** complete
**Priority:** Critical
**Lead:** Unassigned
**Dependencies:** Task 116 (Gateway Basic Connectivity), ADR 058 (Gateway Decoupling)
**Related Documents:** ADR 058 (Gateway Decoupling), Task 116, Task 117, docker-compose.yml

---

## 1. Objective

Perform Red Team security analysis on gateway integration patterns and formally decide on the architecture for connecting Project Sanctuary's 10 script-based MCP servers to the external gateway.

## 2. Deliverables

1. Task 118 documentation file
2. ADR 060: Gateway Integration Patterns
3. Red Team security analysis for each pattern
4. Baseline test verification output
5. Implementation plan for Fleet pattern

## 3. Acceptance Criteria

- Red Team Analysis document completed with security assessment of all 3 patterns
- ADR 060 created with formal decision on integration pattern
- Baseline gateway health verified (3/3 tests passing)
- Clear implementation roadmap for chosen pattern (Fleet)
- Risk mitigation strategies documented for rejected patterns

## Notes

**Critical Decision Point:**
This analysis determines how we connect 10 script-based MCP servers to the gateway without violating ADR 058's decoupling mandate.
**Patterns Under Review:**
1. Trojan Horse (Volume Mounting) - SECURITY RISK
2. Bridge (SSH/Docker Exec) - COMPLEXITY RISK  
3. Fleet (Containerized Sidecars) - RECOMMENDED
**Success Criteria:**
- Maintains ADR 058 isolation principles
- Scalable to 100+ servers
- No dependency hell
- Clear security boundaries

**Status Change (2025-12-17):** backlog → complete
Red Team Analysis completed. Identified Orchestration Fatigue as critical flaw. Pivoted to Hybrid Fleet (Rule of 4). ADR 060 updated with new strategy.
