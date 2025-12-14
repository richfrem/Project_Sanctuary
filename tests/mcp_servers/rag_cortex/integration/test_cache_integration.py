#!/usr/bin/env python3
"""
Integration tests for RAG Cortex cache operations.

Tests cache operations that interact with ChromaDB:
- cache_warmup (populates from ChromaDB)
- guardian_wakeup (populates from ChromaDB)
- cache_get/set/stats (pure memory, but tested here for completeness)
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from mcp_servers.rag_cortex.operations import CortexOperations


def test_cache_operations():
    """
    Test all cache operations in proper sequence.
    
    Sequence:
    1. cache_set - Store answer in cache (pure memory)
    2. cache_get - Retrieve from cache (pure memory)
    3. cache_warmup - Pre-populate cache with genesis queries (queries ChromaDB)
    4. guardian_wakeup - Generate boot digest (queries ChromaDB)
    """
    print("\n" + "="*60)
    print("RAG Cortex Cache Operations Integration Test")
    print("="*60)
    
    ops = CortexOperations(str(project_root))
    
    # Test 1: cache_set (pure memory)
    print("\n[1/4] Testing cache_set...")
    test_query = "What is the meaning of life?"
    test_answer = "42"
    result = ops.cache_set(test_query, test_answer)
    assert result.status == "success"
    assert result.stored is True
    print(f"✓ Cached query: '{test_query}' -> '{test_answer}'")
    print(f"  Cache key: {result.cache_key}")
    
    # Test 2: cache_get (pure memory - should retrieve what we just set)
    print("\n[2/4] Testing cache_get...")
    cached = ops.cache_get(test_query)
    assert cached.cache_hit is True
    assert cached.answer == test_answer
    print(f"✓ Cache hit: '{cached.answer}'")
    print(f"  Query time: {cached.query_time_ms:.2f}ms")
    
    # Test 3: cache_warmup (queries ChromaDB to populate cache)
    print("\n[3/4] Testing cache_warmup...")
    print("  This operation queries ChromaDB to pre-populate cache...")
    warmup_result = ops.cache_warmup()
    assert warmup_result.status == "success"
    print(f"✓ Cache warmed: {warmup_result.queries_cached} queries cached")
    print(f"  Cache hits: {warmup_result.cache_hits}")
    print(f"  Cache misses: {warmup_result.cache_misses}")
    print(f"  Total time: {warmup_result.total_time_ms:.2f}ms")
    
    # Test 4: guardian_wakeup (v2 - Context Synthesis Engine)
    print("\n[4/4] Testing guardian_wakeup (v2 - HOLISTIC mode)...")
    print("  This operation generates a structured intelligence briefing...")
    wakeup_result = ops.guardian_wakeup(mode="HOLISTIC")
    assert wakeup_result.status == "success"
    assert wakeup_result.digest_path is not None
    print(f"✓ Guardian briefing generated: {wakeup_result.digest_path}")
    print(f"  Virtual bundles: {', '.join(wakeup_result.bundles_loaded)}")
    print(f"  Total time: {wakeup_result.total_time_ms:.2f}ms")
    
    print("\n" + "="*60)
    print("✅ ALL CACHE TESTS PASSED (4/4)")
    print("="*60)
    print("\nOperation Summary:")
    print("  • cache_set/cache_get: Pure memory (no ChromaDB)")
    print("  • cache_warmup: Query ChromaDB to populate cache")
    print("  • guardian_wakeup (v2): Context Synthesis Engine (Strategic/Tactical/Recency)")


if __name__ == "__main__":
    test_cache_operations()
