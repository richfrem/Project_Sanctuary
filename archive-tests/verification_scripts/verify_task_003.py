
import sys
import json
import time
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from mcp_servers.cognitive.cortex.server import cortex_ops

def test_caching():
    print("--- Starting Mnemonic Cache Verification ---")
    
    query = "What is the purpose of the Mnemonic Cortex?"
    
    # 1. First Query (Cache Miss)
    print(f"\n1. Executing Query (Expect Miss): '{query}'")
    start = time.time()
    response1 = cortex_ops.query(query, use_cache=True)
    duration1 = time.time() - start
    print(f"   Duration: {duration1:.4f}s")
    print(f"   Cache Hit: {response1.cache_hit}")
    
    if response1.cache_hit:
        print("   [FAIL] Expected cache miss, got hit.")
        return
        
    # 2. Second Query (Cache Hit)
    print(f"\n2. Executing Same Query (Expect Hit): '{query}'")
    start = time.time()
    response2 = cortex_ops.query(query, use_cache=True)
    duration2 = time.time() - start
    print(f"   Duration: {duration2:.4f}s")
    print(f"   Cache Hit: {response2.cache_hit}")
    
    if not response2.cache_hit:
        print("   [FAIL] Expected cache hit, got miss.")
        return
        
    if duration2 > duration1:
        print("   [WARN] Cache hit was slower than miss (cold start overhead?).")
    else:
        print(f"   [SUCCESS] Speedup: {duration1/duration2:.2f}x")

    # 3. Check Stats
    print("\n3. Checking Cache Stats")
    stats = cortex_ops.get_cache_stats()
    print(f"   Stats: {json.dumps(stats, indent=2)}")
    
    if stats.get('hot_cache_size', 0) > 0:
        print("   [SUCCESS] Cache populated.")
    else:
        print("   [FAIL] Cache empty.")

if __name__ == "__main__":
    test_caching()
