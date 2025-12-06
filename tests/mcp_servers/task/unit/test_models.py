import pytest
from mcp_servers.task.models import TaskStatus, TaskPriority

class TestTaskModels:
    def test_task_status_values(self):
        """Verify standard task status values."""
        assert TaskStatus.BACKLOG == "backlog"
        assert TaskStatus.TODO == "todo"
        assert TaskStatus.IN_PROGRESS == "in-progress"
        assert TaskStatus.COMPLETE == "complete"
        assert TaskStatus.BLOCKED == "blocked"

    def test_task_priority_values(self):
        """Verify standard task priority values."""
        assert TaskPriority.LOW == "Low"
        assert TaskPriority.MEDIUM == "Medium"
        assert TaskPriority.HIGH == "High"
        assert TaskPriority.CRITICAL == "Critical"
