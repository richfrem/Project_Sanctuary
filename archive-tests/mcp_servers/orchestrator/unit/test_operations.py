"""
Unit tests for Orchestrator Operations (Business Logic).
Decoupled from Pydantic Models. Mocks file I/O and external calls.
"""
import pytest
import tempfile
import shutil
import os
import json
from unittest.mock import MagicMock, patch
from pathlib import Path
from mcp_servers.orchestrator.operations import OrchestratorOperations

class TestOrchestratorOperations:
    @pytest.fixture
    def setup_ops(self):
        self.test_dir = tempfile.mkdtemp()
        self.ops = OrchestratorOperations(project_root=self.test_dir)
        # Mock validator to avoid config loading issues and enforce isolated paths
        self.ops.validator = MagicMock()
        self.ops.validator.config = {"orchestrator": {"command_file_path": "command.json"}}
        self.ops.config = self.ops.validator.config
        
        # Setup validator to always return valid
        valid_res = MagicMock()
        valid_res.valid = True
        self.ops.validator.validate_path.return_value = valid_res
        self.ops.validator.validate_cognitive_task.return_value = valid_res
        self.ops.validator.validate_git_operation.return_value = valid_res

        yield self.ops
        shutil.rmtree(self.test_dir)

    def test_create_cognitive_task(self, setup_ops):
        """Test generating command.json for cognitive task."""
        res = setup_ops.create_cognitive_task(
            description="Think about life",
            output_path="thought.md",
            max_rounds=3
        )
        
        assert res["status"] == "success"
        assert os.path.exists(res["command_file"])
        
        with open(res["command_file"]) as f:
            data = json.load(f)
            assert data["task_description"] == "Think about life"
            assert data["config"]["max_rounds"] == 3

    def test_run_strategic_cycle_success(self, setup_ops):
        """Test strategic cycle with mocked Cortex."""
        # We need to mock CortexOperations which is imported INSIDE the method
        with patch("mcp_servers.rag_cortex.operations.CortexOperations") as MockCortex:
            # Setup mock instance
            mock_cortex_instance = MockCortex.return_value
            mock_cortex_instance.ingest_incremental.return_value = {"processed": 1}
            mock_cortex_instance.guardian_wakeup.return_value = "Woke up"
            
            report_path = os.path.join(setup_ops.project_root, "report.md")
            with open(report_path, "w") as f: f.write("report")

            result = setup_ops.run_strategic_cycle(
                gap_description="Missing testing",
                research_report_path=report_path
            )
            
            assert "Strategic Crucible Cycle" in result
            assert "Ingestion Complete" in result
            assert "Cache Updated" in result
            
            # Verify ingest called
            mock_cortex_instance.ingest_incremental.assert_called_once()

    def test_get_orchestrator_status_offline(self, setup_ops):
        """Test status when dir is missing."""
        status = setup_ops.get_orchestrator_status()
        assert status["status"] == "offline"

    def test_get_orchestrator_status_online(self, setup_ops):
        """Test status when dir exists."""
        (Path(setup_ops.project_root) / "council_orchestrator").mkdir()
        status = setup_ops.get_orchestrator_status()
        assert status["status"] == "online"

    def test_create_git_commit_task(self, setup_ops):
        """Test git commit task creation with manifest."""
        # Create a dummy file to hash
        fpath = "test.py"
        with open(os.path.join(setup_ops.project_root, fpath), "w") as f:
            f.write("print('hello')")
            
        res = setup_ops.create_git_commit_task(
            files=[fpath],
            message="Initial commit",
            description="Commit work"
        )
        
        assert res["status"] == "success"
        
        with open(res["command_file"]) as f:
            data = json.load(f)
            assert data["task_type"] == "git_commit"
            # Verify manifest hash exists
            assert fpath in data["git_operations"]["p101_manifest"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
