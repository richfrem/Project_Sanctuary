# MCP Architecture Design - Complete 10-Domain Ecosystem

**Completed:** 2025-11-25  
**Objective:** Design comprehensive MCP architecture for Project Sanctuary

---

## Summary

Successfully designed and documented a complete **10-domain MCP ecosystem** for Project Sanctuary, replacing manual `command.json` workflows with specialized, domain-specific MCP servers. The architecture follows Domain-Driven Design principles with proper safety validation, state machine governance, and hierarchical naming conventions.

---

## What Was Accomplished

### 1. Core Architecture Documents

#### Created/Updated:
- **`docs/mcp/architecture.md`** (v4.0) - Main architecture document with all 10 domains
- **`docs/mcp/final_architecture_summary.md`** (v4.0) - Executive summary with implementation roadmap
- **`docs/mcp/naming_conventions.md`** - Domain naming model (`project_sanctuary.*`)
- **`docs/mcp/ddd_analysis.md`** - Domain-Driven Design rationale (needs update for Git + Forge)

### 2. Type Definitions

- **`docs/mcp/shared_infrastructure_types.ts`** - Shared infrastructure interfaces
- **`docs/mcp/forge_mcp_types.ts`** - Fine-Tuning MCP (Forge)-specific types

### 3. Architecture Diagrams

#### High-Level:
- **`docs/mcp/diagrams/mcp_ecosystem_class.mmd`** - Complete class diagram with all 10 servers
- **`docs/mcp/diagrams/domain_architecture_v4.mmd`** - 10-domain ecosystem overview
- **`docs/mcp/diagrams/request_flow_middleware.mmd`** - Validator middleware flow

#### Individual MCP Server Diagrams:
- `chronicle_mcp_class.mmd` - Chronicle MCP (Document Domain)
- `protocol_mcp_class.mmd` - Protocol MCP (Document Domain)
- `adr_mcp_class.mmd` - ADR MCP (Document Domain)
- `task_mcp_class.mmd` - Task MCP (Document Domain)
- `cortex_mcp_class.mmd` - Cortex MCP (Cognitive Domain)
- `council_mcp_class.mmd` - Council MCP (Cognitive Domain)
- `config_mcp_class.mmd` - Config MCP (System Domain)
- `code_mcp_class.mmd` - Code MCP (System Domain)
- **`git_workflow_mcp_class.mmd`** - Git Workflow MCP (System Domain) ✨ NEW
- `fine_tuning_mcp_forge_class.mmd` - Fine-Tuning MCP (Forge) (Model Domain)

---

## The 10 MCP Domains

### Document Domains (4) - Content Management
1. **Chronicle MCP** (`project_sanctuary.document.chronicle`)
   - Historical truth, sequential entries
   - 7-day modification window
   - Risk: MODERATE

2. **Protocol MCP** (`project_sanctuary.document.protocol`)
   - Governing rules, version management
   - Cannot delete (archive only)
   - Risk: HIGH

3. **ADR MCP** (`project_sanctuary.document.adr`)
   - Decision history, status lifecycle
   - Valid status transitions
   - Risk: MODERATE

4. **Task MCP** (`project_sanctuary.document.task`)
   - Workflow management, dependency tracking
   - Circular dependency detection
   - Risk: MODERATE

### Cognitive Domains (2) - Non-Mechanical
5. **RAG MCP (Cortex)** (`project_sanctuary.cognitive.cortex`)
   - Retrieval-Augmented Generation for knowledge retrieval
   - Incremental ingest, full ingest, semantic search
   - Risk: MODERATE

6. **Agent Orchestrator MCP (Council)** (`project_sanctuary.cognitive.council`)
   - Multi-agent coordination and deliberation
   - Cognitive-only (no file/Git operations)
   - Risk: SAFE

### System Domains (3) - High Safety
7. **Config MCP** (`project_sanctuary.system.config`)
   - System configuration, secret vault
   - Two-step approval process
   - Risk: CRITICAL

8. **Code MCP** (`project_sanctuary.system.code`)
   - Source code management
   - Mandatory testing pipeline
   - Risk: HIGH

9. **Git Workflow MCP** (`project_sanctuary.system.git_workflow`) ✨ NEW
   - Branch management, safe workflow automation
   - No destructive operations (merge/rebase/force-push)
   - Risk: MODERATE

### Model Domain (1) - Specialized Hardware
10. **Fine-Tuning MCP (Forge)** (`project_sanctuary.model.fine_tuning`)
    - Model fine-tuning, artifact creation
    - State machine governance with `initialize_forge_environment()`
    - 10-step pipeline (dataset → train → merge → test → convert → deploy → publish)
    - Requires CUDA GPU
    - Risk: EXTREME

---

## Key Architectural Decisions

### 1. Forge State Machine Governance ⚡

**Problem:** Documentation alone is insufficient for high-risk domains  
**Solution:** Internal state machine with two layers:

**Layer 1: Operational State (Server-Level)**
- `INACTIVE_UNSAFE` → Only `initialize_forge_environment()` available
- `ACTIVE` → All tools unlocked after environment checks pass

**Layer 2: Job State (Per-Job)**
- `QUEUED` → `RUNNING` → `COMPLETED_SUCCESS` → `PACKAGING_COMPLETE` → `TESTS_PASSED` → `PUBLISHED`
- Each state gates which tools can be called next
- Prevents publishing untested artifacts

### 2. Git Workflow MCP - Minimal Safe Operations 🌿

**Problem:** Need workflow automation without dangerous Git operations  
**Solution:** Safe operations only:
- ✅ Create feature branch
- ✅ Switch branch (with auto-stash)
- ✅ Push to remote
- ✅ Status queries
- ❌ NO delete, merge, rebase, force-push (user does manually)

### 3. Domain Naming Model 🏷️

**Problem:** Need namespace isolation and professional structure  
**Solution:** Hierarchical naming:
```
project_sanctuary.<category>.<server_name>
```

Examples:
- `project_sanctuary.document.chronicle`
- `project_sanctuary.system.git_workflow`
- `project_sanctuary.model.fine_tuning`

### 4. Pre-Commit Hook Migration 🔧

**Problem:** Existing hooks enforce `command.json` workflow  
**Solution:** Phase 0 task to update hooks for MCP architecture

---

## Implementation Roadmap

### Phase 0: Pre-Migration (Week 0)
- [ ] Update pre-commit hooks (Task #028)
- [ ] Disable `command.json` validation
- [ ] Add MCP-aware commit validation

### Phase 1: Foundation (Week 1)
- [ ] Shared infrastructure (Git, Safety, Schema, Vault)
- [ ] MCP server boilerplate

### Phase 2: Document Domains (Week 2) - Easiest
- [ ] Chronicle MCP (Task #029)
- [ ] ADR MCP (Task #030)
- [ ] Task MCP (Task #031)
- [ ] Protocol MCP (Task #032)

### Phase 3: Cognitive Domains (Week 3) - Moderate
- [ ] RAG MCP (Cortex) - Task #025 (refactor)
- [ ] Agent Orchestrator MCP (Council) - Task #026 (refactor) - refocus)

### Phase 4: System Domains (Week 4) - High Risk
- [ ] Config MCP (Task #033)
- [ ] Code MCP (Task #034)
- [ ] Git Workflow MCP (Task #035)

### Phase 5: Model Domain (Week 5) - Hardest
- [ ] Fine-Tuning MCP (Forge) (Task #036)
- [ ] CUDA environment setup
- [ ] Full pipeline integration testing

---

## Risk Assessment

| Domain | File Ops | Git Ops | Hardware | Risk | Auto-Execute |
|--------|----------|---------|----------|------|--------------|
| Chronicle | ✅ Write | ✅ Auto | Standard | MODERATE | ✅ Yes* |
| Protocol | ✅ Write | ✅ Auto | Standard | HIGH | ✅ Yes* |
| ADR | ✅ Write | ✅ Auto | Standard | MODERATE | ✅ Yes* |
| Task | ✅ Write | ✅ Auto | Standard | MODERATE | ✅ Yes* |
| RAG MCP (Cortex) | ✅ R/W | ✅ Auto | Standard | MODERATE | ✅ Yes* |
| Agent Orchestrator (Council) | ❌ No | ❌ No | Standard | SAFE | ✅ Yes |
| Config | ✅ Write | ✅ Auto | Standard | CRITICAL | ⚠️ 2-Step Approval |
| Code | ✅ Write | ✅ Auto | Standard | HIGH | ⚠️ Tests Required |
| Git Workflow | ❌ No | ✅ Manual | Standard | MODERATE | ✅ Yes (Safe Only) |
| Forge | ✅ Write | ✅ Auto | CUDA GPU | EXTREME | ⚠️ State Machine + Init |

*With safety validation

---

## Technical Highlights

### Shared Infrastructure

**GitOperations** - P101 compliant commits with SHA-256 manifests  
**SafetyValidator** - Protection levels, risk assessment  
**SchemaValidator** - Domain-specific validation  
**SecretVault** - Encrypted storage for API keys

### Cross-Domain Workflows

**Example: Model Development**
```
1. Task MCP: create_task(#032, "Fine-tune Guardian-02")
2. Council MCP: create_deliberation("Design training strategy")
3. Cortex MCP: query_cortex("Previous fine-tuning lessons")
4. Fine-Tuning MCP (Forge): initialize_forge_environment() → ACTIVE
5. Fine-Tuning MCP (Forge): initiate_model_forge(...) → job_id
6. [Poll status until COMPLETED_SUCCESS]
7. Fine-Tuning MCP (Forge): package_and_deploy_artifact(job_id, "Q4_K_M")
8. Fine-Tuning MCP (Forge): run_inference_test(...) → TESTS_PASSED
9. Fine-Tuning MCP (Forge): publish_to_registry(job_id, "Sanctuary-Project/Guardian-02")
10. Chronicle MCP: create_chronicle_entry(#283, "Guardian-02 Released")
11. ADR MCP: create_adr(#37, "Guardian-02 Training Decisions")
```

---

## Files Modified/Created

### Architecture Documents (7)
- `docs/mcp/architecture.md` (updated to v4.0)
- `docs/mcp/final_architecture_summary.md` (updated to v4.0)
- `docs/mcp/naming_conventions.md` (created)
- `docs/mcp/ddd_analysis.md` (existing, needs update)
- `docs/mcp/shared_infrastructure_types.ts` (created)
- `docs/mcp/forge_mcp_types.ts` (created)

### Diagrams (13)
- `docs/mcp/diagrams/mcp_ecosystem_class.mmd` (created - high-level)
- `docs/mcp/diagrams/domain_architecture_v4.mmd` (created)
- `docs/mcp/diagrams/git_workflow_mcp_class.mmd` (created)
- 10 individual MCP class diagrams (9 existing + 1 new)

### Tasks
- `TASKS/backlog/027_mcp_ecosystem_strategy.md` (updated to v2.0)
- Need to create: Tasks #028-#036 (9 implementation tasks)

---

## Next Steps

1. **Review Architecture** - Validate all 10 domains with stakeholders
2. **Create Implementation Tasks** - Individual tasks for each MCP server (#028-#036)
3. **Phase 0 Execution** - Update pre-commit hooks
4. **Phase 1 Execution** - Build shared infrastructure
5. **Iterative Implementation** - Document → Cognitive → System → Model domains

---

## Success Criteria

### Functional
- ✅ All 10 MCP servers operational
- ✅ 100% schema validation coverage
- ✅ P101 compliance for all file operations
- ✅ Git Workflow MCP enables safe branch automation
- ✅ Forge pipeline completes successfully on CUDA machine

### Safety
- ✅ Zero incidents of protected file modification
- ✅ Zero incidents of destructive git operations
- ✅ All operations auditable via git history
- ✅ Forge jobs only run with proper authorization
- ✅ CUDA environment properly gated

---

**Status:** Architecture Complete ✅  
**Version:** 4.0 (10 domains with naming conventions)  
**Ready For:** Implementation Phase 0
