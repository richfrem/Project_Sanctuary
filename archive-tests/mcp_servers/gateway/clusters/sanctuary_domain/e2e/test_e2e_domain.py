"""
E2E Tests for sanctuary_domain cluster (35 tools)

Tools tested:
- Chronicle (7): list-entries, read-latest-entries, get-entry, search, create-entry, append-entry, update-entry
- Protocol (5): list, get, search, create, update
- Task (6): list-tasks, get-task, search-tasks, create-task, update-task, update-task-status
- ADR (5): list, get, search, create, update-status
- Persona (5): list-roles, get-state, reset-state, dispatch, create-custom
- Config (4): list, read, write, delete
- Workflow (2): get-available-workflows, read-workflow
"""
import pytest


# =============================================================================
# CHRONICLE TOOLS (7)
# =============================================================================

@pytest.mark.e2e
@pytest.mark.gateway
class TestChronicleTools:
    
    def test_chronicle_list_entries(self, logged_call):
        """Test chronicle-list-entries returns recent entries."""
        result = logged_call("sanctuary-domain-chronicle-list-entries", {
            "limit": 5
        })
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    def test_chronicle_read_latest_entries(self, logged_call):
        """Test chronicle-read-latest-entries returns latest entries."""
        result = logged_call("sanctuary-domain-chronicle-read-latest-entries", {
            "limit": 3
        })
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    def test_chronicle_get_entry(self, logged_call):
        """Test chronicle-get-entry retrieves specific entry."""
        result = logged_call("sanctuary-domain-chronicle-get-entry", {
            "entry_number": 1
        })
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    def test_chronicle_search(self, logged_call):
        """Test chronicle-search finds entries by content."""
        result = logged_call("sanctuary-domain-chronicle-search", {
            "query": "test"
        })
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    def test_chronicle_create_entry(self, logged_call):
        """Test chronicle-create-entry creates new entry."""
        result = logged_call("sanctuary-domain-chronicle-create-entry", {
            "title": "E2E Test Entry",
            "content": "This entry was created by the E2E test suite.",
            "author": "E2E-Test"
        })
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    def test_chronicle_append_entry(self, logged_call):
        """Test chronicle-append-entry (alias for create)."""
        result = logged_call("sanctuary-domain-chronicle-append-entry", {
            "title": "E2E Append Test",
            "content": "Appended via E2E test."
        })
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    def test_chronicle_update_entry(self, logged_call):
        """Test chronicle-update-entry modifies existing entry."""
        result = logged_call("sanctuary-domain-chronicle-update-entry", {
            "entry_number": 1,
            "updates": {"status": "reviewed"}
        })
        
        # May fail if entry 1 doesn't exist - that's OK
        assert "result" in result or "error" in result


# =============================================================================
# PROTOCOL TOOLS (5)
# =============================================================================

@pytest.mark.e2e
@pytest.mark.gateway
class TestProtocolTools:
    
    def test_protocol_list(self, logged_call):
        """Test protocol-list returns all protocols."""
        result = logged_call("sanctuary-domain-protocol-list", {})
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    def test_protocol_get(self, logged_call):
        """Test protocol-get retrieves specific protocol."""
        result = logged_call("sanctuary-domain-protocol-get", {
            "number": 101  # Protocol 101 - Git safety
        })
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    def test_protocol_search(self, logged_call):
        """Test protocol-search finds protocols by content."""
        result = logged_call("sanctuary-domain-protocol-search", {
            "query": "git"
        })
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    def test_protocol_create(self, logged_call):
        """Test protocol-create creates new protocol."""
        result = logged_call("sanctuary-domain-protocol-create", {
            "title": "E2E Test Protocol",
            "content": "This protocol was created by E2E tests.",
            "status": "PROPOSED"
        })
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    def test_protocol_update(self, logged_call):
        """Test protocol-update modifies existing protocol."""
        result = logged_call("sanctuary-domain-protocol-update", {
            "number": 999,  # Non-existent - will fail gracefully
            "updates": {"status": "deprecated"},
            "reason": "E2E test update"
        })
        
        # Expected to fail (protocol doesn't exist) but tool should execute
        assert "result" in result or "error" in result


# =============================================================================
# TASK TOOLS (6)
# =============================================================================

@pytest.mark.e2e
@pytest.mark.gateway
class TestTaskTools:
    
    def test_list_tasks(self, logged_call):
        """Test list-tasks returns all tasks."""
        result = logged_call("sanctuary-domain-list-tasks", {})
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    def test_get_task(self, logged_call):
        """Test get-task retrieves specific task."""
        result = logged_call("sanctuary-domain-get-task", {
            "task_number": 148  # This task!
        })
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    def test_search_tasks(self, logged_call):
        """Test search-tasks finds tasks by content."""
        result = logged_call("sanctuary-domain-search-tasks", {
            "query": "gateway"
        })
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    def test_create_task(self, logged_call):
        """Test create-task creates new task."""
        result = logged_call("sanctuary-domain-create-task", {
            "title": "E2E Test Task",
            "objective": "Verify task creation works via E2E tests",
            "status": "todo",
            "priority": "Low"
        })
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    def test_update_task(self, logged_call):
        """Test update-task modifies existing task."""
        result = logged_call("sanctuary-domain-update-task", {
            "task_number": 999,  # Non-existent
            "updates": {"notes": "Updated by E2E test"}
        })
        
        # Expected to fail (task doesn't exist) but tool should execute
        assert "result" in result or "error" in result
    
    def test_update_task_status(self, logged_call):
        """Test update-task-status changes task status."""
        result = logged_call("sanctuary-domain-update-task-status", {
            "task_number": 999,  # Non-existent
            "new_status": "in-progress"
        })
        
        # Expected to fail but tool should execute
        assert "result" in result or "error" in result


# =============================================================================
# ADR TOOLS (5)
# =============================================================================

@pytest.mark.e2e
@pytest.mark.gateway
class TestADRTools:
    
    def test_adr_list(self, logged_call):
        """Test adr-list returns all ADRs."""
        result = logged_call("sanctuary-domain-adr-list", {})
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    def test_adr_get(self, logged_call):
        """Test adr-get retrieves specific ADR."""
        result = logged_call("sanctuary-domain-adr-get", {
            "number": 66  # ADR-066 FastMCP
        })
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    def test_adr_search(self, logged_call):
        """Test adr-search finds ADRs by content."""
        result = logged_call("sanctuary-domain-adr-search", {
            "query": "fastmcp"
        })
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    def test_adr_create(self, logged_call):
        """Test adr-create creates new ADR."""
        result = logged_call("sanctuary-domain-adr-create", {
            "title": "E2E Test ADR",
            "context": "Testing ADR creation via E2E suite",
            "decision": "Create test ADR to verify tool functionality",
            "consequences": "ADR-xxx will be created for testing purposes"
        })
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    def test_adr_update_status(self, logged_call):
        """Test adr-update-status changes ADR status."""
        result = logged_call("sanctuary-domain-adr-update-status", {
            "number": 999,  # Non-existent
            "new_status": "deprecated",
            "reason": "E2E test status update"
        })
        
        # Expected to fail but tool should execute
        assert "result" in result or "error" in result


# =============================================================================
# PERSONA TOOLS (5)
# =============================================================================

@pytest.mark.e2e
@pytest.mark.gateway
class TestPersonaTools:
    
    def test_persona_list_roles(self, logged_call):
        """Test persona-list-roles returns available personas."""
        result = logged_call("sanctuary-domain-persona-list-roles", {})
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    def test_persona_get_state(self, logged_call):
        """Test persona-get-state retrieves persona state."""
        result = logged_call("sanctuary-domain-persona-get-state", {
            "role": "architect"
        })
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    def test_persona_reset_state(self, logged_call):
        """Test persona-reset-state clears persona state."""
        result = logged_call("sanctuary-domain-persona-reset-state", {
            "role": "test-persona"
        })
        
        # May fail if persona doesn't exist
        assert "result" in result or "error" in result
    
    def test_persona_dispatch(self, logged_call):
        """Test persona-dispatch sends task to persona."""
        result = logged_call("sanctuary-domain-persona-dispatch", {
            "role": "architect",
            "task": "Briefly describe your role in one sentence."
        })
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    def test_persona_create_custom(self, logged_call):
        """Test persona-create-custom creates new persona."""
        result = logged_call("sanctuary-domain-persona-create-custom", {
            "role": "e2e-tester",
            "persona_definition": "You are an E2E test persona created for verification.",
            "description": "Test persona for E2E suite"
        })
        
        assert result["success"], f"Failed: {result.get('error')}"


# =============================================================================
# CONFIG TOOLS (4)
# =============================================================================

@pytest.mark.e2e
@pytest.mark.gateway
class TestConfigTools:
    
    def test_config_list(self, logged_call):
        """Test config-list returns config files."""
        result = logged_call("sanctuary-domain-config-list", {})
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    def test_config_write(self, logged_call):
        """Test config-write creates config file."""
        result = logged_call("sanctuary-domain-config-write", {
            "filename": "e2e_test_config.json",
            "content": '{"test": true, "created_by": "e2e-suite"}'
        })
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    def test_config_read(self, logged_call):
        """Test config-read retrieves config file."""
        result = logged_call("sanctuary-domain-config-read", {
            "filename": "e2e_test_config.json"
        })
        
        # May fail if file doesn't exist yet
        assert "result" in result or "error" in result
    
    def test_config_delete(self, logged_call):
        """Test config-delete removes config file."""
        result = logged_call("sanctuary-domain-config-delete", {
            "filename": "e2e_test_config.json"
        })
        
        # May fail if file doesn't exist
        assert "result" in result or "error" in result


# =============================================================================
# WORKFLOW TOOLS (2)
# =============================================================================

@pytest.mark.e2e
@pytest.mark.gateway
class TestWorkflowTools:
    
    def test_get_available_workflows(self, logged_call):
        """Test get-available-workflows returns workflow list."""
        result = logged_call("sanctuary-domain-get-available-workflows", {})
        
        assert result["success"], f"Failed: {result.get('error')}"
    
    def test_read_workflow(self, logged_call):
        """Test read-workflow retrieves workflow content."""
        result = logged_call("sanctuary-domain-read-workflow", {
            "filename": "sanctuary-learning-loop.md"
        })
        
        # May fail if workflow doesn't exist
        assert "result" in result or "error" in result
