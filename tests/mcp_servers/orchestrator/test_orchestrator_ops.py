"""
Unit tests for Orchestrator MCP operations.

Tests the query, cognitive, and mechanical tools used by the Orchestrator MCP server.
"""
import pytest
import json
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the tools we're testing
from mcp_servers.orchestrator.tools.query import (
    get_orchestrator_status,
    list_recent_tasks,
    get_task_result
)
from mcp_servers.orchestrator.tools.cognitive import (
    create_cognitive_task,
    create_development_cycle,
    query_mnemonic_cortex
)
from mcp_servers.orchestrator.tools.mechanical import (
    create_file_write_task,
    create_git_commit_task
)


class TestOrchestratorQueryOperations:
    """Test query operations (status, list tasks, get results)."""
    
    @pytest.fixture
    def temp_orchestrator_dir(self, tmp_path):
        """Create temporary orchestrator directory structure."""
        orchestrator_dir = tmp_path / "council_orchestrator"
        orchestrator_dir.mkdir()
        results_dir = orchestrator_dir / "command_results"
        results_dir.mkdir()
        return tmp_path, results_dir
    
    def test_get_status_online(self, temp_orchestrator_dir):
        """Test get_orchestrator_status when directory exists."""
        project_root, _ = temp_orchestrator_dir
        
        status = get_orchestrator_status(project_root=str(project_root))
        
        assert status["status"] == "online"
        assert status["healthy"] is True
        assert "directory" in status
    
    def test_get_status_offline(self, tmp_path):
        """Test get_orchestrator_status when directory doesn't exist."""
        status = get_orchestrator_status(project_root=str(tmp_path))
        
        assert status["status"] == "offline"
        assert status["healthy"] is False
        assert "not found" in status["message"]
    
    def test_list_recent_tasks_empty(self, temp_orchestrator_dir):
        """Test list_recent_tasks with no tasks."""
        project_root, _ = temp_orchestrator_dir
        
        tasks = list_recent_tasks(project_root=str(project_root))
        
        assert isinstance(tasks, list)
        assert len(tasks) == 0
    
    def test_list_recent_tasks_with_results(self, temp_orchestrator_dir):
        """Test list_recent_tasks with task results."""
        project_root, results_dir = temp_orchestrator_dir
        
        # Create mock task result files
        task1 = results_dir / "task_001.json"
        task1.write_text(json.dumps({
            "summary": "Test task 1",
            "status": "completed"
        }))
        
        task2 = results_dir / "task_002.json"
        task2.write_text(json.dumps({
            "summary": "Test task 2",
            "status": "in_progress"
        }))
        
        tasks = list_recent_tasks(limit=10, project_root=str(project_root))
        
        assert len(tasks) == 2
        assert all("task_id" in task for task in tasks)
        assert all("summary" in task for task in tasks)
        assert all("status" in task for task in tasks)
    
    def test_list_recent_tasks_respects_limit(self, temp_orchestrator_dir):
        """Test that list_recent_tasks respects the limit parameter."""
        project_root, results_dir = temp_orchestrator_dir
        
        # Create 5 task files
        for i in range(5):
            task_file = results_dir / f"task_{i:03d}.json"
            task_file.write_text(json.dumps({"summary": f"Task {i}", "status": "completed"}))
        
        tasks = list_recent_tasks(limit=3, project_root=str(project_root))
        
        assert len(tasks) == 3
    
    def test_get_task_result_success(self, temp_orchestrator_dir):
        """Test get_task_result for existing task."""
        project_root, results_dir = temp_orchestrator_dir
        
        # Create a task result
        task_data = {
            "task_id": "test_task",
            "summary": "Test task result",
            "status": "completed",
            "result": "Success"
        }
        task_file = results_dir / "test_task.json"
        task_file.write_text(json.dumps(task_data))
        
        result = get_task_result(task_id="test_task", project_root=str(project_root))
        
        assert result["status"] == "completed"
        assert result["summary"] == "Test task result"
        assert result["result"] == "Success"
    
    def test_get_task_result_not_found(self, temp_orchestrator_dir):
        """Test get_task_result for non-existent task."""
        project_root, _ = temp_orchestrator_dir
        
        result = get_task_result(task_id="nonexistent", project_root=str(project_root))
        
        assert result["status"] == "error"
        assert "not found" in result["error"]
    
    def test_get_task_result_with_json_extension(self, temp_orchestrator_dir):
        """Test get_task_result handles .json extension in task_id."""
        project_root, results_dir = temp_orchestrator_dir
        
        task_data = {"status": "completed"}
        task_file = results_dir / "task.json"
        task_file.write_text(json.dumps(task_data))
        
        result = get_task_result(task_id="task.json", project_root=str(project_root))
        
        assert result["status"] == "completed"


class TestOrchestratorCognitiveOperations:
    """Test cognitive task operations (Council deliberation, dev cycles, RAG queries)."""
    
    @pytest.fixture
    def temp_project(self, tmp_path):
        """Create temporary project structure."""
        orchestrator_dir = tmp_path / "council_orchestrator"
        orchestrator_dir.mkdir()
        work_dir = tmp_path / "WORK_IN_PROGRESS"
        work_dir.mkdir()
        return tmp_path
    
    @patch('mcp_servers.orchestrator.tools.cognitive.write_command_file')
    def test_create_cognitive_task_success(self, mock_write, temp_project):
        """Test create_cognitive_task with valid parameters."""
        mock_write.return_value = str(temp_project / "council_orchestrator" / "command.json")
        
        result = create_cognitive_task(
            description="Test deliberation task",
            output_path="WORK_IN_PROGRESS/test_output.md",
            max_rounds=3,
            project_root=str(temp_project)
        )
        
        assert result["status"] == "success"
        assert "command_file" in result
        assert "queued" in result["message"]
        mock_write.assert_called_once()
    
    @patch('mcp_servers.orchestrator.tools.cognitive.write_command_file')
    def test_create_cognitive_task_with_engine(self, mock_write, temp_project):
        """Test create_cognitive_task with force_engine parameter."""
        mock_write.return_value = str(temp_project / "council_orchestrator" / "command.json")
        
        result = create_cognitive_task(
            description="Test task",
            output_path="WORK_IN_PROGRESS/output.md",
            force_engine="gemini",
            project_root=str(temp_project)
        )
        
        assert result["status"] == "success"
        # Verify the command includes force_engine
        call_args = mock_write.call_args[0][0]
        assert call_args["config"]["force_engine"] == "gemini"
    
    @patch('mcp_servers.orchestrator.tools.cognitive.write_command_file')
    def test_create_cognitive_task_with_input_artifacts(self, mock_write, temp_project):
        """Test create_cognitive_task with input artifacts."""
        mock_write.return_value = str(temp_project / "council_orchestrator" / "command.json")
        
        # Create a test input file
        input_file = temp_project / "WORK_IN_PROGRESS" / "input.md"
        input_file.write_text("Test input")
        
        result = create_cognitive_task(
            description="Test task",
            output_path="WORK_IN_PROGRESS/output.md",
            input_artifacts=["WORK_IN_PROGRESS/input.md"],
            project_root=str(temp_project)
        )
        
        assert result["status"] == "success"
        call_args = mock_write.call_args[0][0]
        assert "input_artifacts" in call_args
    
    @patch('mcp_servers.orchestrator.tools.cognitive.write_command_file')
    def test_create_development_cycle_success(self, mock_write, temp_project):
        """Test create_development_cycle with valid parameters."""
        mock_write.return_value = str(temp_project / "council_orchestrator" / "command.json")
        
        result = create_development_cycle(
            description="Build new feature",
            project_name="test_project",
            output_path="WORK_IN_PROGRESS/dev_cycle.md",
            max_rounds=5,
            project_root=str(temp_project)
        )
        
        assert result["status"] == "success"
        assert "command_file" in result
        assert "test_project" in result["message"]
        
        # Verify command structure
        call_args = mock_write.call_args[0][0]
        assert call_args["task_type"] == "development_cycle"
        assert call_args["project_name"] == "test_project"
    
    @patch('mcp_servers.orchestrator.tools.cognitive.write_command_file')
    def test_query_mnemonic_cortex_success(self, mock_write, temp_project):
        """Test query_mnemonic_cortex with valid parameters."""
        mock_write.return_value = str(temp_project / "council_orchestrator" / "command.json")
        
        result = query_mnemonic_cortex(
            query="What is Protocol 101?",
            output_path="WORK_IN_PROGRESS/query_result.md",
            max_results=10,
            project_root=str(temp_project)
        )
        
        assert result["status"] == "success"
        assert "command_file" in result
        assert "queued" in result["message"]
        
        # Verify command structure
        call_args = mock_write.call_args[0][0]
        assert call_args["task_type"] == "rag_query"
        assert call_args["query"] == "What is Protocol 101?"
        assert call_args["config"]["max_results"] == 10


class TestOrchestratorMechanicalOperations:
    """Test mechanical task operations (file writes, git commits)."""
    
    @pytest.fixture
    def temp_project(self, tmp_path):
        """Create temporary project structure."""
        orchestrator_dir = tmp_path / "council_orchestrator"
        orchestrator_dir.mkdir()
        work_dir = tmp_path / "WORK_IN_PROGRESS"
        work_dir.mkdir()
        return tmp_path
    
    @patch('mcp_servers.orchestrator.tools.mechanical.write_command_file')
    def test_create_file_write_task_success(self, mock_write, temp_project):
        """Test create_file_write_task with valid parameters."""
        mock_write.return_value = str(temp_project / "council_orchestrator" / "command.json")
        
        result = create_file_write_task(
            content="Test content",
            output_path="WORK_IN_PROGRESS/test.md",
            description="Write test file",
            project_root=str(temp_project)
        )
        
        assert result["status"] == "success"
        assert "command_file" in result
        
        # Verify command structure
        call_args = mock_write.call_args[0][0]
        assert call_args["task_type"] == "file_write"
        assert call_args["file_operations"]["path"] == "WORK_IN_PROGRESS/test.md"
        assert call_args["file_operations"]["content"] == "Test content"
    
    @patch('mcp_servers.orchestrator.tools.mechanical.write_command_file')
    def test_create_git_commit_task_success(self, mock_write, temp_project):
        """Test create_git_commit_task with valid parameters."""
        mock_write.return_value = str(temp_project / "council_orchestrator" / "command.json")
        
        # Create test files for hashing
        (temp_project / "file1.py").write_text("# File 1")
        (temp_project / "file2.py").write_text("# File 2")
        
        result = create_git_commit_task(
            files=["file1.py", "file2.py"],
            message="feat: add new feature",
            description="Commit new files",
            project_root=str(temp_project)
        )
        
        assert result["status"] == "success"
        assert "command_file" in result
        
        # Verify command structure
        call_args = mock_write.call_args[0][0]
        assert call_args["task_type"] == "git_commit"
        assert call_args["git_operations"]["commit_message"] == "feat: add new feature"
        assert call_args["git_operations"]["files_to_add"] == ["file1.py", "file2.py"]
        assert "p101_manifest" in call_args["git_operations"]


class TestOrchestratorIntegration:
    """Integration tests for orchestrator operations."""
    
    @pytest.fixture
    def full_project_structure(self, tmp_path):
        """Create complete project structure for integration tests."""
        # Create orchestrator directory
        orchestrator_dir = tmp_path / "council_orchestrator"
        orchestrator_dir.mkdir()
        
        # Create results directory
        results_dir = orchestrator_dir / "command_results"
        results_dir.mkdir()
        
        # Create work directory
        work_dir = tmp_path / "WORK_IN_PROGRESS"
        work_dir.mkdir()
        
        return tmp_path
    
    def test_full_workflow_status_to_task_creation(self, full_project_structure):
        """Test complete workflow: check status, create task, verify it exists."""
        project_root = full_project_structure
        
        # 1. Check orchestrator status
        status = get_orchestrator_status(project_root=str(project_root))
        assert status["healthy"] is True
        
        # 2. Verify no tasks initially
        tasks = list_recent_tasks(project_root=str(project_root))
        assert len(tasks) == 0
        
        # 3. Create a task result manually (simulating task completion)
        results_dir = project_root / "council_orchestrator" / "command_results"
        task_file = results_dir / "integration_test.json"
        task_file.write_text(json.dumps({
            "task_id": "integration_test",
            "summary": "Integration test task",
            "status": "completed"
        }))
        
        # 4. List tasks again
        tasks = list_recent_tasks(project_root=str(project_root))
        assert len(tasks) == 1
        assert tasks[0]["task_id"] == "integration_test"
        
        # 5. Get task result
        result = get_task_result(task_id="integration_test", project_root=str(project_root))
        assert result["status"] == "completed"
        assert result["summary"] == "Integration test task"
