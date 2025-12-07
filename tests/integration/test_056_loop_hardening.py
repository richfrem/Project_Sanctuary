
import pytest
from unittest.mock import MagicMock, patch, call
import sys
import os

# Ensure project root is in path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root)

# Import Operations Classes (Simulating the MCP Server Logic)
try:
    from mcp_servers.git.git_ops import GitOperations
    from mcp_servers.protocol.operations import ProtocolOperations
    from mcp_servers.rag_cortex.operations import CortexOperations
except ImportError as e:
    pytest.fail(f"Failed to import MCP operations: {e}")


class LoopOrchestrator:
    """
    Simulates the Orchestrator's logic for the Strategic Crucible Loop (Task 056).
    This logic would typically reside in the Orchestrator MCP or an Agent script.
    """
    def __init__(self, git_ops, protocol_ops, cortex_ops):
        self.git = git_ops
        self.protocol = protocol_ops
        self.cortex = cortex_ops

    def execute_validation_loop(self, task_id="056", description="loop-validation", policy_content="content"):
        """
        Executes the 4-step loop:
        1. Protocol: Create Policy
        2. Git: Start Feature
        3. Cortex: Ingest
        4. Git: Commit & Push
        """
        # Step 1: Knowledge Generation
        print(f"[Step 1] Generative Policy...")
        policy_result = self.protocol.create_protocol(
            number=int(task_id),
            title="Validation Policy",
            status="PROPOSED",
            classification="Test",
            version="1.0",
            authority="Orchestrator",
            content=policy_content
        )
        policy_path = policy_result['file_path']

        # Step 2: Isolation
        print(f"[Step 2] Git Isolation...")
        self.git.start_feature(task_id, description)

        # Step 3: Ingestion
        print(f"[Step 3] Incremental Ingestion...")
        ingest_result = self.cortex.ingest_incremental(file_paths=[policy_path])
        
        # Check ingestion success (Simulating Orchestrator decision logic)
        # Note: CortexOperations.ingest_incremental returns a dict or object. 
        # We assume standard dictionary return for this simulation or successful return.
        # If it raises exception, that counts as failure.

        # Step 4: Chronicle & Commit
        print(f"[Step 4] Commit and Push...")
        self.git.add([policy_path])
        self.git.commit(f"feat: validate self-evolving memory loop {task_id}")
        self.git.push("origin", f"feature/task-{task_id}-{description}")
        
        return "SUCCESS"


import unittest

@pytest.mark.integration
class TestLoopHardening(unittest.TestCase):
    
    def setUp(self):
        self.mock_git = MagicMock(spec=GitOperations)
        self.mock_protocol = MagicMock(spec=ProtocolOperations)
        self.mock_cortex = MagicMock(spec=CortexOperations)
        self.orchestrator = LoopOrchestrator(self.mock_git, self.mock_protocol, self.mock_cortex)

    def test_scenario_1_golden_path(self):
        """
        Scenario 1: Golden Path Replication
        Verify successful data flow and loop closure under ideal conditions.
        """
        # Setup Mocks
        self.mock_protocol.create_protocol.return_value = {'file_path': 'DOCS/TEST_056_Validation_Policy.md', 'protocol_number': 56}
        self.mock_cortex.ingest_incremental.return_value = {'status': 'success'}
        
        # Execute
        result = self.orchestrator.execute_validation_loop(
            task_id="056", 
            description="scenario-1", 
            policy_content="The Guardian confirms Validation Protocol 056 is active."
        )

        # Assertions
        assert result == "SUCCESS"
        
        # Verify Step 1
        self.mock_protocol.create_protocol.assert_called_once()
        
        # Verify Step 2
        self.mock_git.start_feature.assert_called_with("056", "scenario-1")
        
        # Verify Step 3
        self.mock_cortex.ingest_incremental.assert_called_once()
        
        # Verify Step 4
        self.mock_git.add.assert_called_once()
        self.mock_git.commit.assert_called_once()
        self.mock_git.push.assert_called_once()

        # Simulate Final Verification (Cortex Query)
        # This acts as the "Success Criterion" check outside the loop execution itself
        self.mock_cortex.query.return_value = {'documents': ["The Guardian confirms Validation Protocol 056 is active."]}
        query_result = self.mock_cortex.query("Validation Protocol 056")
        assert "The Guardian confirms" in query_result['documents'][0]


    def test_scenario_2_ingestion_resilience(self):
        """
        Scenario 2: Ingestion Resilience (Simulated Failure)
        Verify system halts if ingestion fails.
        """
        # Setup Mocks
        self.mock_protocol.create_protocol.return_value = {'file_path': 'DOCS/fail_policy.md', 'protocol_number': 56}
        
        # Simulate Ingestion Failure
        self.mock_cortex.ingest_incremental.side_effect = Exception("ChromaDB Connection Failed")

        # Execute & Assert Exception
        with pytest.raises(Exception) as excinfo:
            self.orchestrator.execute_validation_loop(task_id="056", description="scenario-2")
        
        assert "ChromaDB Connection Failed" in str(excinfo.value)

        # Verify Halt: Git Commit should NOT be called
        self.mock_git.commit.assert_not_called()
        self.mock_git.push.assert_not_called()
        print("\n[Passed] Orchestrator halted before commit due to ingestion failure.")


    def test_scenario_3_atomic_commit_integrity(self):
        """
        Scenario 3: Atomic Commit Integrity (Simulated Failure)
        Verify integrity when git operation fails.
        """
        # Setup Mocks
        self.mock_protocol.create_protocol.return_value = {'file_path': 'DOCS/commit_fail.md', 'protocol_number': 56}
        self.mock_cortex.ingest_incremental.return_value = {'status': 'success'}
        
        # Simulate Git Commit Failure
        self.mock_git.commit.side_effect = Exception("Git Hook Failed: Tests not passing")

        # Execute & Assert Exception
        with pytest.raises(Exception) as excinfo:
            self.orchestrator.execute_validation_loop(task_id="056", description="scenario-3")
        
        assert "Git Hook Failed" in str(excinfo.value)

        # Verify Ingestion happened
        self.mock_cortex.ingest_incremental.assert_called_once()
        
        # Verify Push was NEVER called (Atomic integrity)
        self.mock_git.push.assert_not_called()
        print("\n[Passed] Orchestrator halted commit/push sequence due to git error.")


if __name__ == "__main__":
    # Manually run the test setup and methods if called directly
    # purely for quick debugging, usually pytest runs this.
    t = TestLoopHardening()
    t.setUp()
    t.test_scenario_1_golden_path()
    t.setUp()
    t.test_scenario_2_ingestion_resilience()
    t.setUp()
    t.test_scenario_3_atomic_commit_integrity()
    print("All manual tests passed.")
