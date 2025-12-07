
import sys
import os
import logging
from pathlib import Path
import json

# Add project root to path
project_root = Path("/Users/richardfremmerlid/Projects/Project_Sanctuary")
sys.path.append(str(project_root))

from mcp_servers.forge_llm.operations import ForgeOperations

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_forge_status():
    print("\n--- Testing Forge LLM Status ---")
    
    # Initialize operations
    ops = ForgeOperations(project_root=str(project_root))
    
    print("Checking model availability...")
    try:
        status = ops.check_model_availability()
        print(f"Status: {json.dumps(status, indent=2)}")
        
        if status.get("status") == "available":
            print("✅ Forge LLM is connected and ready.")
        else:
            print(f"❌ Forge LLM reported status: {status.get('status')}")
            
    except Exception as e:
        print(f"❌ Error checking status: {e}")

if __name__ == "__main__":
    test_forge_status()
