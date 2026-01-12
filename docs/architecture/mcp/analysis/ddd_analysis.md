# MCP Ecosystem - 10 Domain Architecture (DDD Analysis)

> [!NOTE]
> **Historical Document:** This analysis describes an earlier **10-domain architecture**. The architecture has since evolved to **15 Canonical Servers** per ADR 092, adding Learning (P128), Evolution (P131), Workflow, and additional servers.

**Version:** 3.0 (Legacy)  
**Created:** 2025-11-25  
**Purpose:** Domain-Driven Design analysis of Project Sanctuary MCP ecosystem  
**Status:** Superseded by ADR 092

---

## Executive Summary

Based on Domain-Driven Design (DDD) principles, the Project Sanctuary MCP ecosystem consists of **10 specialized domain servers**, each representing a distinct **Bounded Context** with unique data models, operations, and safety requirements.

---

## Domain Classification

### A. Document Domains (Content Management Bounded Contexts)

These domains share similar toolsets (CRUD, Git, Schema validation) but manage entirely different data types and lifecycles.

| # | Domain | Directory | Core Purpose | Unique Characteristic |
|---|--------|-----------|--------------|----------------------|
| 1 | **Chronicle MCP** | `00_CHRONICLE/` | Historical Truth | Sequential, canonical, rarely-modified entries |
| 2 | **Protocol MCP** | `01_PROTOCOLS/` | Governing Rules | Versioning, formal review, status transitions |
| 3 | **ADR MCP** | `ADRs/` | Decision History | Problem/solution pairs, supersession tracking |
| 4 | **Task MCP** | `tasks/` | Execution Planning | Workflow state transitions, dependency management |

**Shared Characteristics:**
- Markdown-based content
- Git operations with P101 compliance
- Schema validation
- Read/write operations

**Key Differences:**
- **Chronicle**: Immutability focus (7-day modification window)
- **Protocol**: Version management (canonical requires version bump)
- **ADR**: Status lifecycle (Proposed → Accepted → Superseded)
- **Task**: File movement across directories (backlog → active → completed)

---

### B. Cognitive Domains (Non-Mechanical Bounded Contexts)

These domains involve computation/reasoning without direct file system manipulation.

| # | Domain | Directory | Core Purpose | Unique Characteristic |
|---|--------|-----------|--------------|----------------------|
| 5 | **RAG MCP** (Cortex) | `mnemonic_cortex/` | Knowledge Retrieval | RAG operations, incremental/full ingest |
| 6 | **Agent Orchestrator MCP** (Council) | `council_orchestrator/` | Multi-Agent Coordination | Deliberation, NO file/git ops |

**Shared Characteristics:**
- Cognitive/computational focus
- Safety and schema validation
- No direct git operations

**Key Differences:**
- **Cortex**: Data ingestion and retrieval (RAG database)
- **Council**: Command generation for orchestrator (client relationship)

---

### C. System Domains (High-Safety Critical Bounded Contexts)

These domains manage system-critical resources requiring the highest level of safety and governance.

| # | Domain | Directory | Core Purpose | Unique Characteristic |
|---|--------|-----------|--------------|----------------------|
| 7 | **Config MCP** | `.agent/config/` | System Configuration | Two-step approval, secret vault integration |
| 8 | **Code MCP** | `src/`, `scripts/`, `tools/` | Source Code Management | Mandatory testing pipeline, sandbox execution |
| 9 | **Git Workflow MCP** | `.git/` | Branch Management | Safe operations only, no destructive commands |

**Shared Characteristics:**
- Highest safety requirements
- Complex validation pipelines
- Separate audit trails

**Key Differences:**
- **Config**: Sensitive data (secrets never in Git, vault storage)
- **Code**: Executable code (mandatory tests, linting, sandbox execution)
- **Git Workflow**: Branch automation (create, switch, push only - no merge/rebase/delete)

---

### D. Model Domain (Specialized Hardware Bounded Context)

This domain requires specialized hardware and has extreme safety requirements.

| # | Domain | Directory | Core Purpose | Unique Characteristic |
|---|--------|-----------|--------------|----------------------|
| 10 | **Fine-Tuning MCP** (Forge) | `forge/` | LLM Fine-Tuning | State machine governance, CUDA GPU required |

**Unique Characteristics:**
- Requires CUDA-enabled GPU hardware
- State machine with initialization gating
- 10-step pipeline enforcement
- Highest risk level (EXTREME)
- Task MCP authorization required for all jobs

---

## DDD Rationale: Why 10 Domains?

### Why Config MCP is Essential

**Unique Data Model:**
- Configuration files (.json, .yaml, .toml) have specific schemas distinct from Markdown documents
- Mix of public config (committed to Git) and secrets (vault only)
- Hierarchical structure with categories and inheritance

**High Safety Requirements:**
- Changes directly impact system behavior (LLM prompts, agent IDs)
- Security implications (API keys, access lists)
- Requires two-step approval or explicit override
- Separate audited Git flow

**Operations Not Suitable for Other Domains:**
- **Not Chronicle**: Config changes are operational, not historical narrative
- **Not Protocol**: Config is mutable system state, not canonical doctrine
- **Not Task**: Config management is ongoing, not project-based

### Why Code MCP is Essential

**Unique Data Model:**
- Source code files (.py, .js, .sh) with syntax and execution semantics
- Complex dependencies and import graphs
- Test files and test results

**Highest Risk Level:**
- Executable code can modify system behavior
- Bugs can cause data loss or security vulnerabilities
- Requires complex validation (syntax, linting, testing, dependencies)

**Operations Not Suitable for Other Domains:**
- **Not Task**: Code changes require technical validation, not just workflow tracking
- **Not Protocol**: Code is implementation, not specification
- **Too risky for generic Document domains**: Needs dedicated safety pipeline

**Critical Safety Pipeline:**
1. Syntax validation
2. Linter checks (black, flake8)
3. Unit tests (pytest)
4. Dependency verification
5. Safety audit
6. Git commit (only if all pass)

---

## Bounded Context Map

```
┌─────────────────────────────────────────────────────────────┐
│                    LLM Assistant Layer                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  MCP Protocol Interface                      │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Document    │    │  Cognitive   │    │   System     │
│  Domains     │    │  Domains     │    │   Domains    │
│  (4)         │    │  (2)         │    │   (3)        │
│              │    │              │    │              │
│ • Chronicle  │    │ • RAG        │    │ • Config     │
│ • Protocol   │    │   (Cortex)   │    │ • Code       │
│ • ADR        │    │ • Agent Orch │    │ • Git        │
│ • Task       │    │   (Council)  │    │   Workflow   │
└──────────────┘    └──────────────┘    └──────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                              ▼
                    ┌──────────────┐
                    │    Model     │
                    │   Domain     │
                    │    (1)       │
                    │              │
                    │ • Forge      │
                    │   (CUDA)     │
                    └──────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Shared Infrastructure Layer                     │
│  • Git Operations (P101)  • Safety Validator                │
│  • Schema Validator       • Secret Vault                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Safety Level Hierarchy

| Risk Level | Domains | Characteristics | Approval Required |
|------------|---------|-----------------|-------------------|
| **SAFE** | Agent Orchestrator (Council) | Read-only cognitive, no file ops | No |
| **MODERATE** | Chronicle, ADR, Task, RAG (Cortex), Git Workflow | Standard validation, git ops | No (with validation) |
| **HIGH** | Protocol, Code | Version management, mandatory testing | Sometimes (protected protocols/code) |
| **CRITICAL** | Config | System configuration, secrets | Yes (two-step approval) |
| **EXTREME** | Forge | Model training, CUDA hardware | Yes (state machine + task authorization) |

---

## Implementation Priority

### Phase 1: Foundation (Week 1)
- Shared infrastructure (Git, Safety, Schema, Vault)
- MCP server boilerplate

### Phase 2: Document Domains (Week 2)
- Chronicle MCP
- Protocol MCP
- Task MCP
- ADR MCP

### Phase 3: Cognitive Domains (Week 3)
- RAG MCP (Cortex) - Task #025 (refactor)
- Agent Orchestrator MCP (Council) - Task #026 (refactor)

### Phase 4: System Domains (Week 4)
- Config MCP (highest priority for security)
- Code MCP (highest complexity)
- Git Workflow MCP (safe operations only)

### Phase 5: Model Domain (Week 5)
- Fine-Tuning MCP (Forge) - CUDA environment required

---

## Cross-Domain Workflows

### Example 1: Feature Development
```
1. Task MCP: create_task(#028, "Implement Config MCP")
2. Agent Orchestrator MCP (Council): create_deliberation("Design Config MCP architecture")
3. Code MCP: create_script("config_mcp_server.py")
4. Code MCP: run_unit_tests("tests/test_config_mcp.py")
5. Chronicle MCP: create_chronicle_entry(#280, "Config MCP Completed")
6. ADR MCP: create_adr(#36, "Config MCP Architecture Decision")
```

### Example 2: Configuration Update
```
1. Config MCP: get_setting("llm.temperature")
2. Config MCP: update_setting("llm.temperature", 0.7, "Improve creativity", approval_id="GUARDIAN-02")
3. Config MCP: backup_config() [automatic before change]
4. Chronicle MCP: create_chronicle_entry(#281, "LLM Temperature Updated")
```

### Example 3: Code Change with Safety
```
1. Code MCP: create_script("tools/new_utility.py", content, "python")
2. Code MCP: lint_code("tools/new_utility.py") [automatic]
3. Code MCP: run_unit_tests("tests/test_new_utility.py") [mandatory]
4. Code MCP: audit_code_changes(commit_hash) [automatic]
5. Git commit [only if all checks pass]
6. Chronicle MCP: create_chronicle_entry(#282, "New Utility Added")
```

---

## Conclusion

The **10-domain architecture** provides:

1. **Clear Separation of Concerns**: Each domain has a single, well-defined responsibility
2. **Appropriate Safety Levels**: Risk management tailored to each domain's criticality (SAFE → EXTREME)
3. **Maintainability**: Changes to one domain don't affect others
4. **Composability**: Domains work together for complex workflows
5. **DDD Compliance**: Each domain represents a true Bounded Context with unique data models and operations
6. **Hardware Specialization**: Forge domain isolated for CUDA-specific operations
7. **Accessibility**: Generic AI terminology (RAG, Agent Orchestrator) for external developers
8. **Single Responsibility Principle**: Document MCPs handle file operations only; Git Workflow MCP handles all commits

### Separation of Concerns Pattern

**Document MCPs** (Chronicle, Protocol, ADR, Task):
- Create/modify files only
- Return `FileOperationResult` with file paths
- No Git operations

**Git Workflow MCP**:
- Handles all Git commits
- Generates P101 manifests
- Centralizes version control logic

**Benefits:**
- Better composability (LLM chains operations)
- Easier testing (file ops separate from Git)
- More flexible workflows (batch commits)
- Centralized Git logic

**Next Steps:**
1. Finalize shared infrastructure specifications
2. Begin implementation with Document domains (lowest risk)
3. Progress through Cognitive and System domains
4. Complete with Model domain (highest complexity, specialized hardware)

---

**Status:** Architecture Approved - Ready for Implementation  
**Version:** 3.0 (10 Domains)  
**Last Updated:** 2025-11-25
