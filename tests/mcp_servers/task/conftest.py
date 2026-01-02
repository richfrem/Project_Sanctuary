import pytest
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

@pytest.fixture
def task_root(tmp_path):
    """Create a temporary directory for Task tests."""
    root = tmp_path / "task_test_root"
    root.mkdir()
    
    # Create required subdirs
    tasks_dir = root / "tasks"
    tasks_dir.mkdir()
    (tasks_dir / "backlog").mkdir()
    (tasks_dir / "todo").mkdir()
    (tasks_dir / "in-progress").mkdir()
    (tasks_dir / "done").mkdir()
    (tasks_dir / "blocked").mkdir()
    
    return root

@pytest.fixture
def mock_project_root(task_root):
    """Return the temporary root as the project root."""
    return task_root
