# MCP Documentation Organization Plan

## Current State Analysis

After creating `docs/mcp/servers/<name>/` subdirectories, we need to organize the remaining files in `docs/mcp/`.

## Recommendation: File Organization

### ✅ KEEP AT ROOT (`docs/mcp/`)
**Ecosystem-wide documentation that applies to all MCPs or the MCP system as a whole**

| File | Reason to Keep at Root |
|------|------------------------|
| `README.md` | Main entry point for MCP documentation |
| `architecture.md` | 12-domain architecture overview |
| `final_architecture_summary.md` | High-level architecture summary |
| `mcp_operations_inventory.md` | Comprehensive inventory of ALL MCPs |
| `ddd_analysis.md` | Domain-Driven Design analysis (ecosystem-wide) |
| `DOCUMENTATION_STANDARDS.md` | Standards for all MCP documentation |
| `TESTING_STANDARDS.md` | Testing standards for all MCPs |
| `QUICKSTART.md` | Quick start guide for the MCP ecosystem |
| `naming_conventions.md` | Naming conventions across all MCPs |
| `prerequisites.md` | Prerequisites for MCP development |
| `setup_guide.md` | Setup guide for MCP ecosystem |
| `port_registry.md` | Port registry for all MCP servers |
| `claude_desktop_config_template.json` | MCP configuration template |
| `mcp_config_sanctuary.json` | Sanctuary MCP configuration |
| `RAG_STRATEGIES.md` | RAG strategies (ecosystem-wide) |
| `diagrams/` | Ecosystem-wide diagrams |
| `templates/` | Templates for all MCPs |
| `analysis/` | Ecosystem-wide analysis |

### 📁 MOVE TO `servers/council/`
**Council-specific orchestration and testing documentation**

| File | Destination | Reason |
|------|-------------|--------|
| ~~`council_vs_orchestrator.md`~~ | ✅ Already moved | Council/Orchestrator relationship |
| ~~`orchestration_workflows.md`~~ | ✅ Already moved | Council orchestration patterns |
| ~~`mcp_orchestration_validation.md`~~ | ✅ Already moved | Council validation |
| ~~`simple_orchestration_test.md`~~ | ✅ Already moved | Council test scenarios |
| ~~`complete_orchestration_test.md`~~ | ✅ Already moved | Council comprehensive tests |
| ~~`final_orchestration_test.md`~~ | ✅ Already moved | Council end-to-end validation |

### 📁 MOVE TO `servers/rag_cortex/`
**RAG Cortex-specific documentation**

| File | Destination | Reason |
|------|-------------|--------|
| ~~`cortex_evolution.md`~~ | ✅ Already moved | Cortex architecture evolution |
| ~~`cortex_vision.md`~~ | ✅ Already moved | Cortex long-term vision |
| ~~`cortex_operations.md`~~ | ✅ Already moved | Cortex operation specs |
| ~~`cortex_migration_plan.md`~~ | ✅ Already moved | Cortex migration from legacy |
| ~~`cortex_gap_analysis.md`~~ | ✅ Already moved | Cortex feature gaps |
| ~~`cortex_gap_analysis_comprehensive.md`~~ | ✅ Already moved | Cortex detailed gaps |
| ~~`cortex/`~~ | ✅ Already moved to `analysis/` | Cortex analysis files |

### 📁 MOVE TO `servers/forge_llm/`
**Forge LLM-specific documentation**

| File | Destination | Reason |
|------|-------------|--------|
| `forge_mcp_types.ts` | `servers/forge_llm/` | Forge TypeScript types |

### 📁 MOVE TO `servers/orchestrator/`
**Orchestrator-specific testing (if any remain)**

| File | Destination | Reason |
|------|-------------|--------|
| `ollama_direct_test.md` | `servers/orchestrator/` or DELETE | Ollama testing (may be obsolete) |

### ❓ EVALUATE
**Files that may need review**

| File | Recommendation |
|------|----------------|
| `shared_infrastructure_types.ts` | Keep at root (shared across MCPs) |

## Proposed Final Structure

```
docs/mcp/
├── README.md                              (ecosystem entry point)
├── architecture.md                        (12-domain architecture)
├── final_architecture_summary.md          (architecture summary)
├── mcp_operations_inventory.md            (all MCPs inventory)
├── ddd_analysis.md                        (DDD analysis)
├── DOCUMENTATION_STANDARDS.md             (standards)
├── TESTING_STANDARDS.md                   (testing standards)
├── QUICKSTART.md                          (quick start)
├── naming_conventions.md                  (conventions)
├── prerequisites.md                       (prerequisites)
├── setup_guide.md                         (setup)
├── port_registry.md                       (port registry)
├── RAG_STRATEGIES.md                      (RAG strategies)
├── claude_desktop_config_template.json    (config template)
├── mcp_config_sanctuary.json              (config)
├── shared_infrastructure_types.ts         (shared types)
├── diagrams/                              (ecosystem diagrams)
├── templates/                             (templates)
├── analysis/                              (ecosystem analysis)
└── servers/                               (server-specific docs)
    ├── adr/
    │   └── README.md
    ├── agent_persona/
    │   └── README.md
    ├── chronicle/
    │   └── README.md
    ├── code/
    │   └── README.md
    ├── config/
    │   └── README.md
    ├── council/
    │   ├── README.md
    │   ├── council_vs_orchestrator.md
    │   ├── orchestration_workflows.md
    │   ├── mcp_orchestration_validation.md
    │   ├── simple_orchestration_test.md
    │   ├── complete_orchestration_test.md
    │   └── final_orchestration_test.md
    ├── forge_llm/
    │   ├── README.md
    │   └── forge_mcp_types.ts
    ├── git/
    │   └── README.md
    ├── orchestrator/
    │   ├── README.md
    │   └── ollama_direct_test.md (?)
    ├── protocol/
    │   └── README.md
    ├── rag_cortex/
    │   ├── README.md
    │   ├── cortex_evolution.md
    │   ├── cortex_vision.md
    │   ├── cortex_operations.md
    │   ├── cortex_migration_plan.md
    │   ├── cortex_gap_analysis.md
    │   ├── cortex_gap_analysis_comprehensive.md
    │   └── analysis/
    └── task/
        └── README.md
```

## Benefits of This Organization

1. **Clear Separation:** Ecosystem-wide docs at root, server-specific docs in subfolders
2. **Consistency:** Aligns with `mcp_servers/` and `tests/mcp_servers/` structure
3. **Discoverability:** Easy to find server-specific documentation
4. **Maintainability:** Changes to a server's docs are co-located
5. **Scalability:** Easy to add new servers with their own docs

## Next Steps

1. Move `forge_mcp_types.ts` to `servers/forge_llm/`
2. Evaluate `ollama_direct_test.md` (move or delete)
3. Update any broken links in documentation
4. Create a main `docs/mcp/README.md` with navigation to all server docs

---

**Status:** ✅ All server READMEs created, ready to move remaining files
