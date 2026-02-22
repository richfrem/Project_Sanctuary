import pytest
from mcp_servers.task.models import taskstatus, TaskPriority

class TestTaskModels:
    def test_task_status_values(self):
        """Verify standard task status values."""
        assert taskstatus.BACKLOG == "backlog"
        assert taskstatus.TODO == "todo"
        assert taskstatus.IN_PROGRESS == "in-progress"
        assert taskstatus.COMPLETE == "complete"
        assert taskstatus.BLOCKED == "blocked"

    def test_task_priority_values(self):
        """Verify standard task priority values."""
        assert TaskPriority.LOW == "Low"
        assert TaskPriority.MEDIUM == "Medium"
        assert TaskPriority.HIGH == "High"
        assert TaskPriority.CRITICAL == "Critical"
