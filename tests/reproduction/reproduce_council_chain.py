
import sys
import os
import logging
from pathlib import Path
import json

# Add project root to path
project_root = Path("/Users/richardfremmerlid/Projects/Project_Sanctuary")
sys.path.append(str(project_root))

from mcp_servers.council.council_ops import CouncilOperations

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_council_chain():
    print("\n--- Testing Council Orchestrator Chain ---")
    
    # Initialize operations
    council = CouncilOperations(project_root=project_root)
    
    # Define a task
    task = "Briefly explain the role of the Council in Project Sanctuary."
    
    print(f"Dispatching task: '{task}'")
    print("This involves: Council -> AgentPersona -> Ollama (Sanctuary-Qwen2)")
    print("Expect ~50-60s latency for 3 agents x 1 round...")
    
    # Dispatch with limited rounds to save time
    start_time = time.time()
    result = council.dispatch_task(
        task_description=task,
        max_rounds=1, # Single round for validaton
        model_preference="OLLAMA"
    )
    duration = time.time() - start_time
    
    print(f"\nTime taken: {duration:.2f}s")
    print(f"Status: {result.get('status')}")
    
    if result.get('status') == 'success':
        print(f"Session ID: {result.get('session_id')}")
        print(f"Final Synthesis: {result.get('final_synthesis')[:200]}...")
        
        packets = result.get('packets', [])
        print(f"\nPacket Count: {len(packets)}")
        for p in packets:
            print(f" - [{p['member_id']}] Decision: {p['decision'][:50]}...")
    else:
        print(f"Error: {result.get('error')}")

import time

if __name__ == "__main__":
    test_council_chain()
