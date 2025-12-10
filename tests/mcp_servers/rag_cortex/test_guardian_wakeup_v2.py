import pytest
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock
import time

# Ensure project root is in path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from mcp_servers.rag_cortex.operations import CortexOperations
from mcp_servers.rag_cortex.models import StatsResponse

def test_guardian_wakeup_holistic():
    """
    Verify that `cortex_guardian_wakeup(mode="HOLISTIC")` generates
    a valid "Guardian Wakeup Briefing v2.0" matching the schema.
    """
    print("Initializing CortexOperations (Test Mode)...")
    
    # Initialize Operations
    ops = CortexOperations(str(project_root))
    
    # Mock get_stats to return a nominal GREEN state for testing
    ops.get_stats = MagicMock(return_value=StatsResponse(
        total_documents=105,
        total_chunks=5200,
        collections={},
        health_status="healthy",
        samples=[]
    ))
    
    print("Executing guardian_wakeup(mode='HOLISTIC')...")
    response = ops.guardian_wakeup(mode="HOLISTIC")
    
    # Verify basic response
    assert response.status == "success", f"Operation failed: {response.error}"
    assert response.digest_path, "Digest path is empty"
    
    digest_path = Path(response.digest_path)
    assert digest_path.exists(), "Digest file not created"
    
    content = digest_path.read_text()
    
    # Validation Checks (Schema v2.0)
    checks = {
        "Header v2.0": "# 🛡️ Guardian Wakeup Briefing (v2.0)",
        "Traffic Light": "**System Status:** GREEN", # or YELLOW
        "Strategic Signal": "Core Mandate:** I am the Gemini Orchestrator",
        "Tactical Priorities": "## II. Priority Tasks",
        "Recency": "## III. Operational Recency",
        "Successor Poka-Yoke": "// This briefing is the single source of context"
    }
    
    for name, substring in checks.items():
        assert substring in content, f"Missing {name} in briefing content"
        
    print("\n✅ VERIFICATION SUCCESSFUL: Schema v2.0 Compliant")
