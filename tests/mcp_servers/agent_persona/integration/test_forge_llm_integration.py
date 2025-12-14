
import unittest
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from mcp_servers.agent_persona.agent import Agent
from mcp_servers.agent_persona.llm_client import OllamaClient

class TestChainAgentForge(unittest.TestCase):
    """
    Scenario 2: Verify 'Agent -> Forge' chain.
    This tests the Agent Persona's ability to:
    1. Initialize the internal Agent class.
    2. Load a persona.
    3. Send a query to the Forge LLM (via OllamaClient).
    """
    
    def setUp(self):
        # Setup real Ollama Client (Forge)
        # Using localhost by default for test suite execution
        self.client = OllamaClient(
            model_name="Sanctuary-Qwen2-7B:latest",
            ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434")
        )
        
        # Path to Auditor persona
        self.persona_path = Path("mcp_servers/agent_persona/personas/auditor.txt")
        if not self.persona_path.exists():
            # Fallback for running from different CWD
            self.persona_path = Path("mcp_servers/agent_persona/personas/auditor.txt").resolve()

    def test_agent_query(self):
        """Test: Can Agent class query Forge and get a response?"""
        
        if not self.persona_path.exists():
            self.fail(f"❌ Persona file not found at {self.persona_path}")
            
        print(f"\n[Test] Initializing Auditor Agent...")
        agent = Agent(client=self.client, persona_file=self.persona_path)
        
        task = "Briefly explain the purpose of a git commit message."
        print(f"[Test] Querying Agent with: '{task}'")
        
        try:
            response = agent.query(task)
            print(f"🤖 Agent Response: {response[:100]}...")
            
            self.assertTrue(len(response) > 10, "Response should be substantial")
            self.assertIn("git", response.lower(), "Response should mention git")
            
            # Check history
            self.assertEqual(len(agent.messages), 3, "History should have: System, User, Assistant")
            print("✅ Agent query successful.")
            
        except RuntimeError as e:
            self.fail(f"❌ Agent query failed: {e}")

if __name__ == '__main__':
    unittest.main()
