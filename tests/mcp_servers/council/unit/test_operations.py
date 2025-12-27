"""
Unit tests for Council Operations (Business Logic).
Decoupled from Pydantic Models. Mocks external dependencies.
"""
import pytest
from unittest.mock import MagicMock, patch
from mcp_servers.council.operations import CouncilOperations

class TestCouncilOperations:
    @pytest.fixture
    def setup_ops(self):
        """Setup operations with mocked dependencies."""
        self.ops = CouncilOperations()
        
        # Create mocks
        self.ops.persona_ops = MagicMock()
        self.ops.cortex = MagicMock()
        
        # Override initialization to prevent real imports
        self.ops._initialized = True
        
        return self.ops

    def test_list_agents(self, setup_ops):
        """Test listing agents filters correctly."""
        # Setup mock return
        setup_ops.persona_ops.list_roles.return_value = {
            "built_in": ["coordinator", "strategist", "other_role"]
        }
        
        agents = setup_ops.list_agents()
        
        # Should only find coordinator and strategist (other_role is not council member)
        names = [a["name"] for a in agents]
        assert "coordinator" in names
        assert "strategist" in names
        assert "other_role" not in names
        assert "auditor" not in names # Not in built_in list returned by mock

    def test_dispatch_task_flow(self, setup_ops):
        """Test the multi-round dispatch loop."""
        # Mock RAG
        setup_ops.cortex.query.return_value = {"results": [{"content": "Context", "metadata": {"source": "doc"}}]}
        
        # Mock Persona Dispatch
        setup_ops.persona_ops.dispatch.return_value = {
            "response": "I agree.",
            "reasoning": "Good plan.",
            "engine": "ollama"
        }
        
        result = setup_ops.dispatch_task(
            task_description="Decide usage.",
            max_rounds=2,
            agent="coordinator" # Single agent for simplicity
        )
        
        assert result["status"] == "success"
        assert result["rounds"] == 2
        assert len(result["packets"]) == 2 # 2 rounds * 1 agent
        assert result["final_synthesis"] == "I agree."
        
        # Verify calls
        setup_ops.cortex.query.assert_called_once()
        assert setup_ops.persona_ops.dispatch.call_count == 2

    def test_dispatch_task_no_rag(self, setup_ops):
        """Test dispatch handles RAG failure gracefully."""
        setup_ops.cortex.query.side_effect = Exception("RAG Down")
        setup_ops.persona_ops.dispatch.return_value = {"response": "Ok"}
        
        result = setup_ops.dispatch_task("Quick task", agent="coordinator", max_rounds=1)
        
        assert result["status"] == "success"
        # Processing should continue despite RAG error

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
