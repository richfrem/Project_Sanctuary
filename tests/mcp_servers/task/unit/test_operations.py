"""
Unit tests for Task Operations (Business Logic).
Decoupled from Pydantic Models.
"""
import pytest
import shutil
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from mcp_servers.task.operations import TaskOperations
from mcp_servers.task.models import TaskPriority, TaskStatus

class TestTaskOperations:
    @pytest.fixture
    def setup_ops(self, tmp_path):
        # Create necessary directories
        (tmp_path / "TASKS").mkdir()
        (tmp_path / "TASKS" / "backlog").mkdir()
        (tmp_path / "TASKS" / "todo").mkdir()
        (tmp_path / "TASKS" / "in-progress").mkdir()
        (tmp_path / "TASKS" / "done").mkdir()
        
        ops = TaskOperations(project_root=tmp_path)
        
        # Mock validator to isolate operations logic
        ops.validator = MagicMock()
        
        # Default validator mocks
        ops.validator.validate_task_number.return_value = (True, "")
        ops.validator.validate_task_schema.return_value = (True, [])
        ops.validator.get_next_task_number.return_value = 1
        
        return ops

    def test_create_task(self, setup_ops):
        """Test creating a task file."""
        res = setup_ops.create_task(
            title="Test Task",
            objective="Do things",
            deliverables=["Item 1"],
            acceptance_criteria=["Works"],
            status=TaskStatus.BACKLOG
        )

        assert res.status == "success"
        assert res.task_number == 1
        
        # Verify file creation
        task_dir = setup_ops.tasks_dir / "backlog"
        files = list(task_dir.glob("*.md"))
        assert len(files) == 1
        assert "001_test_task.md" in files[0].name
        
        content = files[0].read_text()
        assert "# TASK: Test Task" in content
        assert "**Status:** backlog" in content

    def test_update_task(self, setup_ops):
        """Test updating a task."""
        # Create initial task
        setup_ops.create_task("Old", "Obj", ["Del"], ["Crit"], status=TaskStatus.BACKLOG)
        
        # Mock task existence for update
        setup_ops.validator.task_exists.return_value = (True, str(setup_ops.tasks_dir / "backlog"))
        
        # Update
        res = setup_ops.update_task(1, {"title": "New Title"})
        
        assert res.status == "success"
        
        # Verify content
        task_file = setup_ops.tasks_dir / "backlog" / "001_initial_task.md" # Note: title changes content, but filename stays unless renamed? 
        # Actually filename generation happens on create. Update uses existing file.
        # But wait, create_task generates filename based on title. 
        # update_task reads file, updates schema, writes back. It does NOT rename file.
        
        # Re-find file
        files = list((setup_ops.tasks_dir / "backlog").glob("001_*.md"))
        content = files[0].read_text()
        assert "# TASK: New Title" in content

    def test_update_task_status_move(self, setup_ops):
        """Test moving task file on status change."""
        # Create in Backlog
        setup_ops.create_task("Move Me", "Obj", ["Del"], ["Crit"], status=TaskStatus.BACKLOG)
        
        # Mock task existence
        setup_ops.validator.task_exists.return_value = (True, str(setup_ops.tasks_dir / "backlog"))
        
        # Move to Todo
        res = setup_ops.update_task_status(1, TaskStatus.TODO, "Moving up")
        
        assert res.status == "success"
        
        # Check old location empty
        assert not list((setup_ops.tasks_dir / "backlog").glob("*.md"))
        
        # Check new location
        new_files = list((setup_ops.tasks_dir / "todo").glob("*.md"))
        assert len(new_files) == 1
        
        # Check content for notes
        content = new_files[0].read_text()
        assert "**Status:** todo" in content
        assert "Moving up" in content

    def test_list_tasks(self, setup_ops):
        """Test listing tasks."""
        # Create manually to avoid logic dependence
        p1 = setup_ops.tasks_dir / "backlog" / "001_one.md"
        p1.write_text("# TASK: One\n**Status:** backlog\n**Priority:** medium\n...", encoding='utf-8')
        
        p2 = setup_ops.tasks_dir / "todo" / "002_two.md"
        p2.write_text("# TASK: Two\n**Status:** todo\n**Priority:** high\n...", encoding='utf-8')

        tasks = setup_ops.list_tasks()
        assert len(tasks) == 2
        assert tasks[0]["number"] == 1
        assert tasks[1]["number"] == 2
        
        # Test filter
        high_pri = setup_ops.list_tasks(priority=TaskPriority.HIGH)
        assert len(high_pri) == 1
        assert high_pri[0]["title"] == "Two"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
