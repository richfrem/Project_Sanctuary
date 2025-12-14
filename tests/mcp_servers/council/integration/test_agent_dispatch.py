o
import unittest
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mcp_servers.council.council_ops import CouncilOperations
from mcp_servers.agent_persona.agent_persona_ops import AgentPersonaOperations

class TestChainCouncilAgent(unittest.TestCase):
    """
    Scenario 3: Verify 'Council -> Agent -> Forge' chain.
    This tests the Council's deliberation logic, ensuring it can:
    1. Initialize.
    2. Dispatch a task to an Auditor agent.
    3. Receive and structure the result.
    """
    
    def setUp(self):
        # We need a mocked Cortex for context retrieval to avoid needing the ChromaDB container
        # for this specific test, OR we just let it fail gracefully (Council handles RAG failure).
        
        # Initialize operations
        self.council_ops = CouncilOperations()
        
        # Mock the cortex client to return dummy context
        # This focuses the test on Counsel -> Agent logic, not RAG logic
        self.council_ops.cortex = MagicMock()
        self.council_ops.cortex.query.return_value = {"results": ["Context 1", "Context 2"]}
        
    def test_single_agent_dispatch(self):
        """Test: Can Council dispatch to a single agent?"""
        print(f"\n[Test] Dispatching 'Auditor' task via Council...")
        
        task = "What is the primary risk of using 'git push --force'?"
        
        # Dispatch to just the Auditor
        result = self.council_ops.dispatch_task(
            task_description=task,
            agent="auditor",
            max_rounds=1,
            model_name="Sanctuary-Qwen2-7B:latest",
            force_engine="ollama" # Use our running Ollama instance
        )
        
        # Verify structure
        self.assertIn("session_id", result)
        self.assertEqual(result["status"], "success")
        self.assertEqual(len(result["packets"]), 1, "Should be 1 round packet")
        
        packet = result["packets"][0]
        decision = packet["decision"]
        
        print(f"🤖 Council/Auditor Decision: {decision[:100]}...")
        
        self.assertIn("overwrite", decision.lower(), "Auditor should mention overwriting history")
        self.assertEqual(packet["member_id"], "auditor")
        
        print("✅ Council dispatch successful.")

if __name__ == '__main__':
    unittest.main()
