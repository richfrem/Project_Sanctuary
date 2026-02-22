
import sys
import os
import logging
from pathlib import Path

# Add project root to path
project_root = Path("/Users/richardfremmerlid/Projects/Project_Sanctuary")
sys.path.append(str(project_root))

from mcp_servers.agent_persona.agent_persona_ops import AgentPersonaOperations
from mcp_servers.agent_persona.llm_client import get_llm_client

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_ollama_direct():
    print("\n--- Testing Ollama Connection Directly ---")
    try:
        import ollama
        client = ollama.Client(host="http://127.0.0.1:11434")
        models = client.list()
        print(f"✅ Connected to Ollama! Found {len(models['models'])} models.")
        for m in models['models']:
            print(f" - {m['name']}")
    except Exception as e:
        print(f"❌ Failed to connect to Ollama: {e}")

def test_persona_dispatch():
    print("\n--- Testing AgentPersonaOperations.dispatch ---")
    ops = AgentPersonaOperations(project_root=project_root)
    
    # Try listing roles first
    roles = ops.list_roles()
    print(f"Roles available: {roles['built_in']}")
    
    # Try dispatching to a simple role
    print("Dispatching to 'coordinator'...")
    result = ops.dispatch(
        role="coordinator",
        task="Say hello and confirm you are online.",
        engine="ollama", # Force ollama
        model_name="Sanctuary-Qwen2-7B:latest", # Use known good model
        maintain_state=False
    )
    
    print(f"Result Status: {result.get('status')}")
    if result.get('status') == 'success':
        print(f"Response: {result.get('response')}")
    else:
        print(f"Error: {result.get('error')}")

if __name__ == "__main__":
    test_ollama_direct()
    test_persona_dispatch()
