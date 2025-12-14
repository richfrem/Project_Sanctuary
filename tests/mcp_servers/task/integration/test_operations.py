"""
Task MCP Integration Tests - Operations Testing
================================================

Tests each Task MCP operation against temp directories with cleanup.
Uses pytest fixtures for isolated test environments.

CALLING EXAMPLES:
-----------------
pytest tests/mcp_servers/task/integration/test_operations.py -v -s
pytest tests/mcp_servers/task/integration/test_operations.py::TestCreateTask -v

MCP OPERATIONS:
---------------
| Operation          | Type  | Description               |
|--------------------|-------|---------------------------|
| create_task        | WRITE | Create new task           |
| get_task           | READ  | Get task by number        |
| list_tasks         | READ  | List tasks with filters   |
| search_tasks       | READ  | Search tasks by content   |
| update_task        | WRITE | Update task metadata      |
| update_task_status | WRITE | Move task between statuses|
"""

import pytest
from pathlib import Path

from mcp_servers.task.operations import TaskOperations
from mcp_servers.task.models import TaskStatus, TaskPriority


@pytest.fixture
def task_ops(task_root):
    """Create TaskOperations instance using shared fixture"""
    from mcp_servers.task.operations import TaskOperations
    
    # Create tools directory with get_next_task_number.py (specific to this test need)
    tools_dir = task_root / "tools" / "scaffolds"
    tools_dir.mkdir(parents=True, exist_ok=True)
    
    # Simple version of get_next_task_number
    (tools_dir / "get_next_task_number.py").write_text("""
def get_next_task_number():
    return "001"
""")

    return TaskOperations(task_root)


class TestCreateTask:
    """Test create_task operation"""
    
    def test_create_task_success(self, task_ops):
        """Test successful task creation"""
        result = task_ops.create_task(
            title="Test Task",
            objective="Test objective",
            deliverables=["Deliverable 1", "Deliverable 2"],
            acceptance_criteria=["Criterion 1", "Criterion 2"],
            priority=TaskPriority.HIGH,
            status=TaskStatus.BACKLOG
        )
        
        assert result.status == "success"
        assert result.operation == "created"
        assert result.task_number > 0  # Just verify a task number was assigned
        assert "TASKS/backlog/" in result.file_path
        assert "_test_task.md" in result.file_path
        assert "# TASK: Test Task" in result.content
    
    def test_create_task_with_dependencies(self, task_ops):
        """Test task creation with dependencies"""
        # Create first task
        task_ops.create_task(
            title="First Task",
            objective="First",
            deliverables=["D1"],
            acceptance_criteria=["C1"],
            task_number=1
        )
        
        # Create second task with dependency
        result = task_ops.create_task(
            title="Second Task",
            objective="Second",
            deliverables=["D2"],
            acceptance_criteria=["C2"],
            dependencies="Requires #001",
            task_number=2
        )
        
        assert result.status == "success"
        assert "Requires #001" in result.content
    
    def test_create_task_duplicate_number(self, task_ops):
        """Test creating task with duplicate number fails"""
        # Create first task
        task_ops.create_task(
            title="First",
            objective="First",
            deliverables=["D1"],
            acceptance_criteria=["C1"],
            task_number=1
        )
        
        # Try to create duplicate
        result = task_ops.create_task(
            title="Duplicate",
            objective="Duplicate",
            deliverables=["D2"],
            acceptance_criteria=["C2"],
            task_number=1
        )
        
        assert result.status == "error"
        assert "already exists" in result.message


class TestUpdateTask:
    """Test update_task operation"""
    
    def test_update_task_priority(self, task_ops):
        """Test updating task priority"""
        # Create task
        task_ops.create_task(
            title="Test",
            objective="Test",
            deliverables=["D1"],
            acceptance_criteria=["C1"],
            priority=TaskPriority.MEDIUM,
            task_number=1
        )
        
        # Update priority
        result = task_ops.update_task(
            task_number=1,
            updates={"priority": TaskPriority.CRITICAL}
        )
        
        assert result.status == "success"
        assert result.operation == "updated"
        assert "Critical" in result.content
    
    def test_update_nonexistent_task(self, task_ops):
        """Test updating non-existent task fails"""
        result = task_ops.update_task(
            task_number=999,
            updates={"priority": TaskPriority.HIGH}
        )
        
        assert result.status == "error"
        assert "not found" in result.message
    
    def test_update_task_with_string_values(self, task_ops):
        """Test updating task with string values (as received from MCP)"""
        # Create task
        task_ops.create_task(
            title="Test",
            objective="Test",
            deliverables=["D1"],
            acceptance_criteria=["C1"],
            priority=TaskPriority.MEDIUM,
            task_number=1
        )
        
        # Update with string values (simulating MCP input)
        result = task_ops.update_task(
            task_number=1,
            updates={
                "priority": "High",
                "lead": "Test User",
                "notes": "Updated via MCP"
            }
        )
        
        assert result.status == "success"
        assert result.operation == "updated"
        assert "High" in result.content
        assert "Test User" in result.content
        assert "Updated via MCP" in result.content
    
    def test_parse_capitalized_status(self, task_ops, task_root):
        """Test parsing task files with capitalized status values"""
        # Create a task file with capitalized status
        task_file = task_root / "TASKS" / "backlog" / "001_test_capitalized.md"
        task_file.write_text("""# TASK: Test Capitalized Status

**Status:** Backlog
**Priority:** High
**Lead:** Test User
**Dependencies:** None
**Related Documents:** None

---

## 1. Objective

Test objective

## 2. Deliverables

1. Deliverable 1

## 3. Acceptance Criteria

- Criterion 1
""")
        
        # Should be able to read and list this task
        tasks = task_ops.list_tasks(status=TaskStatus.BACKLOG)
        assert len(tasks) >= 1
        
        # Should be able to get this task
        task = task_ops.get_task(1)
        assert task is not None
        assert task["status"] == "backlog"


class TestUpdateTaskStatus:
    """Test update_task_status operation"""
    
    def test_move_task_to_in_progress(self, task_ops):
        """Test moving task from backlog to in-progress"""
        # Create task in backlog
        task_ops.create_task(
            title="Test",
            objective="Test",
            deliverables=["D1"],
            acceptance_criteria=["C1"],
            status=TaskStatus.BACKLOG,
            task_number=1
        )
        
        # Move to in-progress
        result = task_ops.update_task_status(
            task_number=1,
            new_status=TaskStatus.IN_PROGRESS,
            notes="Starting work"
        )
        
        assert result.status == "success"
        assert result.operation == "moved"
        assert "in-progress" in result.file_path
        assert "Starting work" in result.content
    
    def test_move_task_to_done(self, task_ops):
        """Test moving task to done"""
        # Create and move task
        task_ops.create_task(
            title="Test",
            objective="Test",
            deliverables=["D1"],
            acceptance_criteria=["C1"],
            task_number=1
        )
        
        result = task_ops.update_task_status(
            task_number=1,
            new_status=TaskStatus.COMPLETE
        )
        
        assert result.status == "success"
        assert "done" in result.file_path
    
    def test_move_task_to_todo(self, task_ops):
        """Test moving task from backlog to todo (as tested in Claude)"""
        # Create task in backlog
        task_ops.create_task(
            title="Test Todo Move",
            objective="Test",
            deliverables=["D1"],
            acceptance_criteria=["C1"],
            status=TaskStatus.BACKLOG,
            task_number=1
        )
        
        # Move to todo
        result = task_ops.update_task_status(
            task_number=1,
            new_status=TaskStatus.TODO
        )
        
        assert result.status == "success"
        assert result.operation == "moved"
        assert "todo" in result.file_path


class TestGetTask:
    """Test get_task operation"""
    
    def test_get_existing_task(self, task_ops):
        """Test retrieving existing task"""
        # Create task
        task_ops.create_task(
            title="Test Task",
            objective="Test",
            deliverables=["D1"],
            acceptance_criteria=["C1"],
            task_number=1
        )
        
        # Get task
        task = task_ops.get_task(1)
        
        assert task is not None
        assert task["number"] == 1
        assert task["title"] == "Test Task"
        assert task["status"] == "backlog"
    
    def test_get_nonexistent_task(self, task_ops):
        """Test retrieving non-existent task returns None"""
        task = task_ops.get_task(999)
        assert task is None


class TestListTasks:
    """Test list_tasks operation"""
    
    def test_list_all_tasks(self, task_ops):
        """Test listing all tasks"""
        # Create multiple tasks
        for i in range(1, 4):
            task_ops.create_task(
                title=f"Task {i}",
                objective="Test",
                deliverables=["D1"],
                acceptance_criteria=["C1"],
                task_number=i
            )
        
        tasks = task_ops.list_tasks()
        assert len(tasks) == 3
    
    def test_list_tasks_by_status(self, task_ops):
        """Test filtering tasks by status"""
        # Create tasks with different statuses
        task_ops.create_task(
            title="Backlog Task",
            objective="Test",
            deliverables=["D1"],
            acceptance_criteria=["C1"],
            status=TaskStatus.BACKLOG,
            task_number=1
        )
        
        task_ops.create_task(
            title="In Progress Task",
            objective="Test",
            deliverables=["D1"],
            acceptance_criteria=["C1"],
            status=TaskStatus.IN_PROGRESS,
            task_number=2
        )
        
        # List only backlog tasks
        backlog_tasks = task_ops.list_tasks(status=TaskStatus.BACKLOG)
        assert len(backlog_tasks) == 1
        assert backlog_tasks[0]["title"] == "Backlog Task"
    
    def test_list_tasks_by_priority(self, task_ops):
        """Test filtering tasks by priority"""
        # Create tasks with different priorities
        task_ops.create_task(
            title="High Priority",
            objective="Test",
            deliverables=["D1"],
            acceptance_criteria=["C1"],
            priority=TaskPriority.HIGH,
            task_number=1
        )
        
        task_ops.create_task(
            title="Low Priority",
            objective="Test",
            deliverables=["D1"],
            acceptance_criteria=["C1"],
            priority=TaskPriority.LOW,
            task_number=2
        )
        
        # List only high priority tasks
        high_tasks = task_ops.list_tasks(priority=TaskPriority.HIGH)
        assert len(high_tasks) == 1
        assert high_tasks[0]["title"] == "High Priority"


class TestSearchTasks:
    """Test search_tasks operation"""
    
    def test_search_by_title(self, task_ops):
        """Test searching tasks by title"""
        # Create tasks
        task_ops.create_task(
            title="Authentication Feature",
            objective="Add auth",
            deliverables=["D1"],
            acceptance_criteria=["C1"],
            task_number=1
        )
        
        task_ops.create_task(
            title="Database Migration",
            objective="Migrate DB",
            deliverables=["D1"],
            acceptance_criteria=["C1"],
            task_number=2
        )
        
        # Search for "authentication"
        results = task_ops.search_tasks("authentication")
        assert len(results) == 1
        assert results[0]["title"] == "Authentication Feature"
    
    def test_search_no_results(self, task_ops):
        """Test search with no matches"""
        task_ops.create_task(
            title="Test",
            objective="Test",
            deliverables=["D1"],
            acceptance_criteria=["C1"],
            task_number=1
        )
        
        results = task_ops.search_tasks("nonexistent")
        assert len(results) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
