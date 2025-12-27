"""
Unit tests for Workflow Operations (Business Logic).
Decoupled from Pydantic Models.
"""
import pytest
import sys
from unittest.mock import MagicMock, patch
from pathlib import Path

# Mock yaml module if not present or to avoid dependency
mock_yaml = MagicMock()
mock_yaml.safe_load.return_value = {"description": "Test Workflow"}

with patch.dict(sys.modules, {"yaml": mock_yaml}):
    from mcp_servers.workflow.operations import WorkflowOperations

class TestWorkflowOperations:
    @pytest.fixture
    def setup_ops(self, tmp_path):
        self.workflow_dir = tmp_path / "workflows"
        self.workflow_dir.mkdir()
        return WorkflowOperations(workflow_dir=self.workflow_dir)

    def test_list_workflows(self, setup_ops):
        """Test listing workflows."""
        # Create dummy workflow
        p1 = setup_ops.workflow_dir / "test_flow.md"
        p1.write_text("---\ndescription: Test Workflow\n---\nSteps...", encoding='utf-8')
        
        # Create another without frontmatter
        p2 = setup_ops.workflow_dir / "simple.md"
        p2.write_text("Just steps", encoding='utf-8')

        # We need to ensure yaml.safe_load is called or mocked correctly inside the class method
        # transforming the content
        with patch("mcp_servers.workflow.operations.yaml") as local_yaml:
            local_yaml.safe_load.side_effect = [{"description": "Test Workflow"}, {}]
            
            workflows = setup_ops.list_workflows()
            
            # Sort order: simple.md, test_flow.md (alphabetical by filename)
            # wait, s comes before t.
            assert len(workflows) == 2
            assert workflows[0]["filename"] == "simple.md"
            assert workflows[1]["filename"] == "test_flow.md"
            
            # Check parsing
            # verification of mocked yaml result
            # logic: _parse_frontmatter uses yaml.safe_load
            assert workflows[1]["description"] == "Test Workflow"

    def test_get_workflow_content(self, setup_ops):
        """Test retrieving content."""
        p1 = setup_ops.workflow_dir / "read_me.md"
        p1.write_text("Content", encoding='utf-8')
        
        content = setup_ops.get_workflow_content("read_me.md")
        assert content == "Content"
        
        missing = setup_ops.get_workflow_content("missing.md")
        assert missing is None

    def test_turbo_mode_detection(self, setup_ops):
        """Test detection of turbo mode flag."""
        p1 = setup_ops.workflow_dir / "turbo.md"
        p1.write_text("---\n---\n// turbo-all\nSteps", encoding='utf-8')
        
        with patch("mcp_servers.workflow.operations.yaml") as local_yaml:
             local_yaml.safe_load.return_value = {}
             workflows = setup_ops.list_workflows()
             assert workflows[0]["turbo_mode"] is True

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
