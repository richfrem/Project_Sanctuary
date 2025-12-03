"""
Integration tests for Cortex operations - following verify_all.py pattern.
"""
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

@pytest.mark.integration
def test_cache_operations():
    """Test cache get/set operations directly."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    
    from mnemonic_cortex.core.cache import MnemonicCache
    cache = MnemonicCache()
    
    # Test Set
    test_key = "integration_test_key"
    test_val = {"status": "verified", "timestamp": "now"}
    cache.set(test_key, test_val)
    
    # Test Get
    retrieved = cache.get(test_key)
    assert retrieved == test_val, f"Cache mismatch. Expected {test_val}, got {retrieved}"
    
    print(f"✅ Cache operations verified: {retrieved}")

@pytest.mark.integration  
def test_guardian_wakeup():
    """Test Guardian Wakeup operation."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    
    from mcp_servers.rag_cortex.operations import CortexOperations
    ops = CortexOperations(str(PROJECT_ROOT))
    result = ops.guardian_wakeup()
    
    assert result.status == "success", f"Guardian wakeup failed: {result}"
    assert result.digest_path is not None, "No digest path returned"
    
    print(f"✅ Guardian Wakeup verified: {result.digest_path}")

@pytest.mark.integration
def test_adaptation_packet_generation():
    """Test Adaptation Packet Generation."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    
    from mnemonic_cortex.app.synthesis.generator import SynthesisGenerator
    gen = SynthesisGenerator(str(PROJECT_ROOT))
    packet = gen.generate_packet(days=1)  # Match verify_all.py
    
    assert packet is not None, "No packet generated"
    # Don't assert examples > 0 - might be 0 if no recent changes (like verify_all.py)
    
    print(f"✅ Adaptation packet generated: {len(packet.examples)} examples")
