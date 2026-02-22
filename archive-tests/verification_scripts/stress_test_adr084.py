import sys
from unittest.mock import MagicMock
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from mcp_servers.rag_cortex.operations import CortexOperations
from mcp_servers.rag_cortex.models import PersistSoulRequest

def test_dead_mans_switch():
    print("🧪 Starting ADR 084 Dead-Man's Switch Stress Test...")
    
    # Initialize Operations
    ops = CortexOperations(project_root)
    
    # Mock calculate_semantic_entropy to fail
    # This simulates the "Dead-Man's Switch" condition
    ops._calculate_semantic_entropy = MagicMock(side_effect=ValueError("Simulated Entropy Calculation Failure"))
    
    # Prepare Request
    req = PersistSoulRequest(
        snapshot_path=".agent/learning/learning_package_snapshot.md",
        valence=0.0,
        uncertainty=0.1
    )
    
    # Execute
    print("   Action: Calling persist_soul with induced volatility...")
    response = ops.persist_soul(req)
    
    # Validation
    # Use loose matching since Global Floor (0.95) catches 1.0 before the explicit Volatile check
    if response.status == "quarantined" and ("Epistemic Gate: VOLATILE" in response.error or "Global Floor breach" in response.error):
         print(f"✅ STATUS: VETOED | REASON: Fail-Closed Gating (SE=1.0)")
         print(f"   Detailed Error: {response.error}")
    else:
         print(f"❌ FAILURE: Response was {response.status}")
         print(f"   Error: {response.error}")
         sys.exit(1)

if __name__ == "__main__":
    test_dead_mans_switch()
