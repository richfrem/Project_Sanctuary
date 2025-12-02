#!/usr/bin/env python3
"""
Manual Verification Script for Multi-Round Deliberation Logic (Task #086B)

This script simulates a full Council deliberation (3 rounds) by mocking the 
LLM and Cortex dependencies but executing the actual Council and Agent Persona 
logic. It prints the conversation flow to verify the logic integrity.
"""

import os
import sys
import json
from unittest.mock import MagicMock, patch

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 1. Mock legacy modules to allow imports
sys.modules["mnemonic_cortex.app.services.vector_db_service"] = MagicMock()
sys.modules["mnemonic_cortex.app.services.llm_service"] = MagicMock()

# 2. Import Operations
from mcp_servers.lib.council.council_ops import CouncilOperations
from mcp_servers.cognitive.cortex.operations import CortexOperations

def run_verification():
    print("🔍 Starting Multi-Round Deliberation Verification...")
    
    # 3. Setup Mocks
    with patch('mcp_servers.lib.agent_persona.agent_persona_ops.get_llm_client') as mock_llm, \
         patch.object(CortexOperations, 'query') as mock_cortex_query, \
         patch.object(CortexOperations, 'cache_warmup'):
        
        # Mock Cortex Context
        mock_cortex_query.return_value = MagicMock(
            results=[
                MagicMock(
                    content="Protocol 87: Structured Queries. Defines how to query the system.",
                    metadata={"source": "01_PROTOCOLS/087.md"},
                    score=0.95
                )
            ]
        )
        
        # Mock LLM Responses (Context-Aware)
        def mock_generate(prompt, **kwargs):
            prompt_lower = prompt.lower()
            
            # Identify Agent
            agent = "unknown"
            if "coordinator" in prompt_lower: agent = "Coordinator"
            elif "strategist" in prompt_lower: agent = "Strategist"
            elif "auditor" in prompt_lower: agent = "Auditor"
            
            # Identify Round (heuristic based on context length or content)
            # This is a simplification; in real flow, context grows.
            if "critique" in prompt_lower or "review" in prompt_lower: 
                phase = "Critique"
            elif "synthesize" in prompt_lower or "final" in prompt_lower:
                phase = "Synthesis"
            else:
                phase = "Initial Proposal"
                
            return f"[{agent} - {phase}] Based on Protocol 87, I suggest..."

        mock_client = MagicMock()
        mock_client.generate.side_effect = mock_generate
        mock_llm.return_value = mock_client
        
        # 4. Execute Council Dispatch
        print("\n🚀 Dispatching Task to Council (Max Rounds: 2)...")
        council_ops = CouncilOperations()
        result = council_ops.dispatch_task(
            task_description="Design a new protocol for MCP composition",
            agent=None, # Full Council
            max_rounds=2
        )
        
        # 5. Analyze Results
        print("\n✅ Execution Complete!")
        print(f"Status: {result['status']}")
        print(f"Rounds Completed: {result['rounds']}")
        print(f"Agents Involved: {len(result['agents'])} ({', '.join(result['agents'])})")
        print(f"Total Packets: {len(result['packets'])}")
        
        print("\n📜 Conversation Flow:")
        current_round = -1
        for packet in result['packets']:
            if packet['round_id'] != current_round:
                current_round = packet['round_id']
                print(f"\n--- ROUND {current_round + 1} ---")
            
            print(f"\n👤 {packet['member_id'].upper()}:")
            print(f"   {packet['decision']}")
            
        print("\n🧠 Final Synthesis:")
        print(f"   {result['final_synthesis']}")
        
        # 6. Verify Logic
        print("\n🕵️ Logic Verification:")
        
        # Check 1: Cortex was queried
        cortex_called = mock_cortex_query.call_count > 0
        print(f"   [{'x' if cortex_called else ' '}] Cortex Context Retrieved")
        
        # Check 2: All agents participated
        all_agents = set(result['agents']) == {'coordinator', 'strategist', 'auditor'}
        print(f"   [{'x' if all_agents else ' '}] All Agents Participated")
        
        # Check 3: Rounds executed
        rounds_ok = result['rounds'] == 2
        print(f"   [{'x' if rounds_ok else ' '}] Multi-Round Execution")
        
        if cortex_called and all_agents and rounds_ok:
            print("\n✅ VERIFICATION SUCCESSFUL: Logic integrity confirmed.")
        else:
            print("\n❌ VERIFICATION FAILED: Logic gaps detected.")

if __name__ == "__main__":
    run_verification()
