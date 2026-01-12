# MCP Ecosystem - Legacy 12-Domain Architecture

> [!NOTE]
> This document describes the **original 12-Domain architecture**. For the current Canonical 15-Server architecture (which adds Learning, Evolution, and Workflow servers), see **ADR 092**.

**Version:** 4.0 (Legacy)  
**Created:** 2025-11-25  
**Status:** Superseded by ADR 092 - Canonical 15 Servers

---

## Complete Domain Map (12 Servers — Legacy)

> See ADR 092 for the current 15-server architecture.

| # | Domain | Category | Directory | Risk Level | Hardware |
|---|--------|----------|-----------|------------|----------|
| 1 | **Chronicle MCP** | Document | `00_CHRONICLE/` | MODERATE | Standard |
| 2 | **Protocol MCP** | Document | `01_PROTOCOLS/` | HIGH | Standard |
| 3 | **ADR MCP** | Document | `ADRs/` | MODERATE | Standard |
| 4 | **Task MCP** | Document | `tasks/` | MODERATE | Standard |
| 5 | **RAG MCP** (Cortex) | Cognitive | `mnemonic_cortex/` | MODERATE | Standard |
| 6 | **Agent Persona MCP** | Cognitive | `mcp_servers/agent_persona/` | SAFE | Standard |
| 7 | **Council MCP** | Cognitive | `mcp_servers/council/` | SAFE | Standard |
| 8 | **Orchestrator MCP** | Cognitive | `mcp_servers/orchestrator/` | SAFE | Standard |
| 9 | **Config MCP** | System | `.agent/config/` | CRITICAL | Standard |
| 10 | **Code MCP** | System | `src/`, `scripts/`, `tools/` | HIGH | Standard |
| 11 | **Git MCP** | System | `.git/` | MODERATE | Standard |
| 12 | **Forge LLM MCP** | Model | `mcp_servers/forge_llm/` | EXTREME | **CUDA GPU** |

---

## Domain Categories

### I. Document Domains (4) - Content Management
**Shared Characteristics:**
- Markdown-based content
- Git operations with P101 compliance
- Schema validation
- CRUD operations

**Individual Focus:**
- **Chronicle**: Historical truth, sequential entries, 7-day modification window
- **Protocol**: Governing rules, version management, canonical status
- **ADR**: Decision history, status lifecycle, supersession tracking
- **Task**: Workflow management, dependency tracking, file movement

---

### II. Cognitive Domains (4) - Non-Mechanical
**Shared Characteristics:**
- Computation/reasoning focus
- No direct file system manipulation
- Safety and schema validation

**Individual Focus:**
- **RAG MCP** (Cortex): Retrieval-Augmented Generation for knowledge retrieval
  - Incremental ingest, full ingest, semantic search
  - Industry-standard RAG pattern with ChromaDB
  - Project implementation: Mnemonic Cortex
- **Agent Persona MCP**: Management of AI personas and roles
  - Create, list, and dispatch to personas
  - State management for agent conversations
- **Council MCP**: Multi-agent deliberation and collaboration
  - Facilitates discussion between multiple agents
  - Uses Agent Persona MCP for execution
- **Orchestrator MCP**: High-level mission planning and execution
  - Strategic cycles and long-running workflows
  - Coordinates other MCPs

---

### III. System Domains (3) - High-Safety Critical
**Shared Characteristics:**
- Highest safety requirements
- Complex validation pipelines
- Separate audit trails

**Individual Focus:**
- **Config**: System configuration, secret vault, two-step approval
- **Code**: Source code management, mandatory testing, linting pipeline
- **Git Workflow**: Branch management, safe workflow automation, read-only by default

---

### IV. Model Domain (1) - Specialized Hardware
**Unique Characteristics:**
- **CUDA GPU requirement**
- Asynchronous job execution
- 10-step model lifecycle pipeline
- Extreme safety validation

**Focus:**
- **Forge**: Model fine-tuning, artifact creation, Hugging Face publishing

---

## Fine-Tuning MCP (Forge): The Model Lifecycle Orchestrator

### Hardware Requirements
- **CUDA-enabled GPU** (validated on RTX A2000)
- **WSL environment** with ml_env activated
- **Sufficient resources**: GPU memory, disk space
- **Environment marker**: `CUDA_FORGE_ACTIVE=true`

### 10-Step Pipeline

| Step | Tool | Script | Purpose |
|------|------|--------|---------|
| 1 | `initiate_model_forge` | `forge_whole_genome_dataset.py` | Create training dataset |
| 2 | ↳ (async) | `fine_tune.py` | Fine-tune model with QLoRA |
| 3 | ↳ (async) | `merge_adapter.py` | Merge LoRA adapter with base |
| 4 | `run_inference_test` | `inference.py` | Test merged model |
| 5 | `package_and_deploy_artifact` | `convert_to_gguf.py` | Convert to GGUF format |
| 6 | ↳ (sync) | `create_modelfile.py` | Generate Ollama Modelfile |
| 7 | ↳ (sync) | `ollama create` | Import to local Ollama |
| 8 | `run_inference_test` | `ollama run` | Test both interaction modes |
| 9 | `publish_to_registry` | `upload_to_huggingface.py` | Upload to Hugging Face |
| 10 | `retrieve_registry_artifact` | Download from HF | Verify upload integrity |

### Safety Rules (Extreme)

**Environment Gate:**
- Must check for `CUDA_FORGE_ACTIVE` marker
- Must verify CUDA availability
- Must confirm ml_env activation

**Resource Reservation:**
- Check GPU memory before starting
- Check disk space for model artifacts
- Reject job if insufficient resources

**Task Linkage:**
- All jobs must link to Task MCP entry
- Provides audit trail and prioritization

**Script Whitelist:**
- Only whitelisted scripts can execute
- No arbitrary `os.system()` or `subprocess.run()`
- Prevents command injection

**Artifact Integrity:**
- SHA-256 validation (P101-style)
- Manifest generation for all artifacts
- Verification before marking complete

---

## Cross-Domain Workflow Example

**Scenario:** Fine-tune Sanctuary-Guardian-02 model

**Workflow (Separation of Concerns Pattern):**

```
1. Task MCP: create_task(#032, "Fine-tune Sanctuary-Guardian-02")
5. Fine-Tuning MCP (Forge): initiate_model_forge({
     forge_id: "guardian-02-v1",
     authorization_task_id: 32,
     hyperparameters: {...}
   }) → returns job_id
6. [Wait for async job completion, poll with get_forge_job_status]
7. Fine-Tuning MCP (Forge): package_and_deploy_artifact(job_id, "Q4_K_M")
8. Fine-Tuning MCP (Forge): run_inference_test(model_path, test_prompts)
9. Fine-Tuning MCP (Forge): publish_to_registry(job_id, "Sanctuary-Project/Guardian-02")
10. Chronicle MCP: create_chronicle_entry(#283, "Guardian-02 Model Released")
11. ADR MCP: create_adr(#37, "Guardian-02 Training Decisions")
12. Task MCP: update_task_status(32, "Completed")
```

---

## Implementation Roadmap

### Phase 0: Pre-Migration (Week 0)
- [ ] Update pre-commit hooks to work with MCP architecture
- [ ] Disable or adapt `command.json` validation hooks
- [ ] Add MCP-aware commit message validation

### Phase 1: Foundation (Week 1)
- [ ] Shared infrastructure (Git, Safety, Schema, Vault)
- [ ] MCP server boilerplate
- [ ] CUDA environment verification module

### Phase 2: Document Domains (Week 2)
- [ ] Chronicle MCP
- [ ] Protocol MCP
- [ ] Task MCP
- [ ] ADR MCP

### Phase 3: Cognitive Domains (Week 3)
- [ ] RAG MCP (Cortex) - Task #025 (refactor)
- [ ] Agent Orchestrator MCP (Council) - Task #026 (refactor)

### Phase 4: System Domains (Week 4)
- [ ] Config MCP (highest security priority)
- [ ] Code MCP (highest complexity)
- [ ] Git Workflow MCP (safe operations only)

### Phase 5: Model Domain (Week 5)
- [ ] Fine-Tuning MCP (Forge) - requires CUDA machine setup
- [ ] Integration testing with full pipeline
- [ ] Documentation and deployment

---

## Risk Assessment Matrix

| Domain | File Ops | Git Ops | Hardware | Risk Level | Auto-Execute |
|--------|----------|---------|----------|------------|--------------|
| Chronicle | ✅ Write | ✅ Auto | Standard | MODERATE | ✅ Yes* |
| Protocol | ✅ Write | ✅ Auto | Standard | HIGH | ✅ Yes* |
| ADR | ✅ Write | ✅ Auto | Standard | MODERATE | ✅ Yes* |
| Task | ✅ Write | ✅ Auto | Standard | MODERATE | ✅ Yes* |
| Cortex | ✅ Read/Write | ✅ Auto | Standard | MODERATE | ✅ Yes* |
| Agent Persona | ❌ No | ❌ No | Standard | SAFE | ✅ Yes |
| Council | ❌ No | ❌ No | Standard | SAFE | ✅ Yes |
| Orchestrator | ❌ No | ❌ No | Standard | SAFE | ✅ Yes |
| Config | ✅ Write | ✅ Auto | Standard | CRITICAL | ⚠️ Approval Required |
| Code | ✅ Write | ✅ Auto | Standard | HIGH | ⚠️ Tests Required |
| Git | ❌ No | ✅ Manual | Standard | MODERATE | ✅ Yes (Safe Ops) |
| Forge LLM | ✅ Write | ✅ Auto | **CUDA GPU** | EXTREME | ⚠️ Resource Check + Approval |

*With safety validation

---

## Architecture Artifacts

All architecture documentation is in `docs/architecture/mcp/`:

**Core Documents:**
- `architecture.md` - Main architecture document (v4.0 - 10 domains)
- `ddd_analysis.md` - DDD rationale for 8 domains (needs update for Git + Forge)
- `final_architecture_summary.md` - This document
- `walkthrough.md` - Complete implementation walkthrough
- `naming_conventions.md` - Domain naming model

**Type Definitions:**
- `shared_infrastructure_types.ts` - Shared infrastructure interfaces
- `forge_mcp_types.ts` - Forge-specific types

**Diagrams:**
- `diagrams/mcp_ecosystem_class.mmd` - **High-level class diagram (all 12 domains)**
- `diagrams/domain_architecture_v3.mmd` - Complete 12-domain ecosystem
- `diagrams/request_flow_middleware.mmd` - Validator middleware flow
- `diagrams/chronicle_mcp_class.mmd` - Chronicle MCP class diagram
- `diagrams/protocol_mcp_class.mmd` - Protocol MCP class diagram
- `diagrams/adr_mcp_class.mmd` - ADR MCP class diagram
- `diagrams/task_mcp_class.mmd` - Task MCP class diagram
- `diagrams/rag_mcp_cortex_class.mmd` - RAG MCP (Cortex) class diagram
- `diagrams/agent_orchestrator_mcp_council_class.mmd` - Agent Orchestrator MCP (Council) class diagram
- `diagrams/config_mcp_class.mmd` - Config MCP class diagram
- `diagrams/code_mcp_class.mmd` - Code MCP class diagram
- `diagrams/git_workflow_mcp_class.mmd` - Git Workflow MCP class diagram
- `diagrams/fine_tuning_mcp_forge_class.mmd` - Fine-Tuning MCP (Forge) class diagram

---

## Success Criteria

### Functional
- [ ] All 15 MCP servers operational
- [ ] 100% schema validation coverage
- [ ] P101 compliance for all file operations
- [ ] Git safety rules enforced
- [ ] Git Workflow MCP enables safe branch automation
- [ ] Forge pipeline completes successfully on CUDA machine

### Safety
- [ ] Zero incidents of protected file modification
- [ ] Zero incidents of destructive git operations
- [ ] All operations auditable via git history
- [ ] Forge jobs only run with proper authorization
- [ ] CUDA environment properly gated

### Performance
- [ ] Sub-second response for read operations
- [ ] Asynchronous job handling for long-running tasks (Forge)
- [ ] Proper resource management (no GPU memory leaks)

---

**Status:** Architecture Complete - Ready for Task #027 Implementation  
**Next Step:** Begin Phase 1 (Shared Infrastructure)  
**Owner:** Guardian (via Gemini 2.0 Flash Thinking Experimental)
