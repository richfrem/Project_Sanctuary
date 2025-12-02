# Task 087: Comprehensive MCP Operations Testing

## Metadata
- **Status**: in-progress
- **Priority**: High
- **Complexity**: High
- **Category**: Testing
- **Estimated Effort**: 8-12 hours
- **Dependencies**: None
- **Created**: 2025-12-01
- **Updated**: 2025-12-01

## Objective

Perform comprehensive testing of all 11 MCP servers after recent changes (logging additions, documentation updates, gap analysis). Verify that all operations work correctly both via test harnesses and through the Antigravity agent interface.

## Deliverables

1. Test harness execution results for all 11 MCPs
2. Antigravity operation testing results for all 11 MCPs
3. Bug reports for any failures
4. Updated test coverage documentation

## Testing Approach

### Phase 1: Test Harness Validation (Run First) 🧪

For each MCP, run the pytest test harness to validate underlying operations:

1. **Chronicle MCP**
   ```bash
   pytest tests/test_chronicle_operations.py tests/test_chronicle_validator.py -v
   ```

2. **Protocol MCP**
   ```bash
   pytest tests/test_protocol_operations.py tests/test_protocol_validator.py -v
   ```

3. **ADR MCP**
   ```bash
   pytest tests/test_adr_operations.py tests/test_adr_validator.py -v
   ```

4. **Task MCP**
   ```bash
   pytest tests/test_task_operations.py tests/test_task_validator.py -v
   ```

5. **Cortex MCP**
   ```bash
   pytest tests/mcp_servers/cortex/ -v
   ```

6. **Agent Persona MCP**
   ```bash
   pytest tests/integration/test_agent_persona_with_cortex.py -v
   ```

7. **Council MCP**
   ```bash
   pytest tests/mcp_servers/council/ -v
   ```

8. **Forge MCP**
   ```bash
   pytest tests/integration/test_forge_model_serving.py -v
   ```

9. **Git Workflow MCP**
   ```bash
   pytest tests/integration/test_council_with_git.py -v
   ```

10. **Code MCP**
    ```bash
    # Check if tests exist
    find tests -name "*code*" -type f
    ```

11. **Config MCP**
    ```bash
    # Check if tests exist
    find tests -name "*config*" -type f
    ```

### Phase 2: Antigravity Operation Testing (MCP Tool Interface) 🤖

For each MCP, test key operations directly via Antigravity:

#### 1. Chronicle MCP
- [ ] `chronicle_create_entry` - Create a test entry
- [ ] `chronicle_list_entries` - List recent entries
- [ ] `chronicle_get_entry` - Retrieve specific entry
- [ ] `chronicle_search` - Search entries

#### 2. Protocol MCP
- [ ] `protocol_create` - Create test protocol
- [ ] `protocol_list` - List protocols
- [ ] `protocol_get` - Retrieve Protocol 101
- [ ] `protocol_search` - Search protocols

#### 3. ADR MCP
- [ ] `adr_create` - Create test ADR
- [ ] `adr_list` - List ADRs
- [ ] `adr_get` - Retrieve specific ADR
- [ ] `adr_update_status` - Update ADR status

#### 4. Task MCP
- [ ] `create_task` - Create test task
- [ ] `list_tasks` - List tasks by status
- [ ] `get_task` - Retrieve specific task
- [ ] `update_task_status` - Update task status

#### 5. Cortex MCP
- [ ] `cortex_query` - Query knowledge base
- [ ] `cortex_ingest_incremental` - Incremental ingestion
- [ ] `cortex_get_stats` - Get collection stats
- [ ] `cortex_query_structured` - Protocol 87 query

#### 6. Agent Persona MCP
- [ ] `persona_dispatch` - Dispatch to coordinator
- [ ] `persona_list_roles` - List available roles
- [ ] `persona_get_state` - Get persona state
- [ ] `persona_create_custom` - Create custom persona

#### 7. Council MCP
- [ ] `council_dispatch` - Full council deliberation
- [ ] `council_dispatch` (single agent) - Specific agent
- [ ] `council_list_agents` - List agents

#### 8. Forge MCP
- [ ] `check_sanctuary_model_status` - Check model availability
- [ ] `query_sanctuary_model` - Query the model

#### 9. Git Workflow MCP
- [ ] `git_status` - Check git status
- [ ] `git_create_branch` - Create feature branch
- [ ] `git_smart_commit` - Smart commit

#### 10. Code MCP
- [ ] `code_read` - Read file
- [ ] `code_write` - Write file
- [ ] `code_search_content` - Search code
- [ ] `code_list_files` - List files

#### 11. Config MCP
- [ ] `config_read` - Read config
- [ ] `config_write` - Write config
- [ ] `config_list` - List configs

## Acceptance Criteria

- [ ] All 11 MCP test harnesses execute successfully
- [ ] All key operations tested via Antigravity
- [ ] Any failures documented with bug reports
- [ ] Test coverage gaps identified
- [ ] Updated `mcp_operations_inventory.md` with test results

## Success Metrics

- **Test Harnesses**: 11/11 passing
- **Antigravity Operations**: All key operations verified
- **Documentation**: Test results recorded in inventory

## Related Documents

- [MCP Operations Inventory](../../docs/mcp/mcp_operations_inventory.md)
- [Integration Tests](../../tests/integration/)
- [Testing Standards](../../docs/mcp/TESTING_STANDARDS.md)
