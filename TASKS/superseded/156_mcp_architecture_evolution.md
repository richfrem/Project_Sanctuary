# Task 156: MCP Architecture Evolution (15 Servers)

**Objective**: Implement the "Canonical 15" architecture defined in ADR 092, separating System Lifecycle (Protocol 128) and Evolutionary (Protocol 131) concerns from the RAG Cortex core.

## 1. Create New MCP Servers
- [ ] **Create `mcp_servers/learning`**
    - [ ] Initialize directory structure (standard template)
    - [ ] Create `server.py` and `operations.py`
    - [ ] Migrate `learning_debrief`, `capture_snapshot`, `persist_soul`, `guardian_wakeup` from `rag_cortex`
    - [ ] Ensure `operations.py` implements Protocol 128 logic independent of Vector DB
- [ ] **Create `mcp_servers/evolution`**
    - [ ] Initialize directory structure
    - [ ] Create `server.py` and `operations.py`
    - [ ] Migrate `metrics.py` (from `LEARNING/topics/drq...`) to `mcp_servers/evolution/metrics.py`
    - [ ] Implement `measure_complexity` wrapper in `operations.py`

## 2. Refactor RAG Cortex
- [ ] **Clean `mcp_servers/rag_cortex`**
    - [ ] Remove migrated operations from `operations.py`
    - [ ] Remove unused imports/dependencies specific to lifecycle logic
    - [ ] Verify `rag_cortex` runs as pure Vector DB service

## 3. Update CLI Tools
- [ ] **Update `scripts/cortex_cli.py`**
    - [ ] Refactor imports to pull from `mcp_servers.learning.operations` for lifecycle commands
    - [ ] Ensure `ingest` and `query` still pull from `mcp_servers.rag_cortex.operations`
    - [ ] (Optional) Rename/Alias to `scripts/learning_cli.py` if needed, or keep `cortex_cli.py` as unified frontend

## 4. Domain & Cluster Updates (Gateway)
- [ ] **Update `mcp_servers/gateway/fleet_spec.py`**
    - [ ] Ensure `sanctuary_cortex` cluster includes the new servers in its mental model (if applicable)
- [ ] **Update `mcp_servers/gateway/clusters/sanctuary_cortex/server.py`**
    - [ ] Register new Lifecycle/Learning tools (pointing to `learning` backend)
    - [ ] Register new Evolution tools (pointing to `evolution` backend)
- [ ] **Update `mcp_servers/gateway/fleet_registry.json`**
    - [ ] Add new tool definitions/capabilities
    - [ ] Ensure documentation descriptions match new locations

## 5. Documentation Update (Comprehensive)
- [ ] **Update `mcp_operations_inventory.md`**
    - [ ] Add `Learning MCP` (14) and `Evolution MCP` (15) sections
    - [ ] Move operations from RAG Cortex section to new sections
    - [ ] Add `Workflow MCP` (13) section
- [ ] **Update `mcp_servers/README.md`**
    - [ ] List all 15 canonical servers
- [ ] **Retroactive Documentation Consistency**
    - [ ] Update `ADRs/034_containerize_mcp_servers_with_podman.md` (Reflect 15 servers)
    - [ ] Update `ADRs/046_standardize_all_mcp_servers_on_fastmcp_implementation.md`
    - [ ] Update `ADRs/047_mandate_live_integration_testing_for_all_mcps.md`
    - [ ] Update `00_CHRONICLE/ENTRIES/327_...` (Contextual references)
- [ ] **Update Architecture Diagrams**
    - [ ] Update `docs/architecture_diagrams` (if applicable)

## 6. Deployment & Cleanup
- [ ] **Rebuild Podman Images**
    - [ ] Ensure new server dependencies are included
- [ ] **Redeploy Fleet**
    - [ ] Restart Gateway
- [ ] **Cleanup**
    - [ ] Delete `mcp_servers/sanctuary_gateway` (erroneous folder)
    - [ ] Delete `LEARNING/topics/drq.../src/metrics.py` (after verifying migration)
