
import sys
import json
import time
from pathlib import Path
from mcp_servers.cognitive.cortex.server import cortex_ops

def test_ingestion():
    print("--- Starting Native Ingestion Verification ---")
    
    # Test Incremental Ingestion (Faster)
    test_file = "mnemonic_cortex/RAG_STRATEGIES_AND_DOCTRINE.md"
    print(f"\n1. Testing Incremental Ingestion of: {test_file}")
    
    project_root = Path(__file__).parent.parent
    sys.path.append(str(project_root))
    start = time.time()
    response = cortex_ops.ingest_incremental(
        file_paths=[test_file],
        skip_duplicates=False # Force re-ingest to test logic
    )
    duration = time.time() - start
    
    print(f"   Duration: {duration:.4f}s")
    print(f"   Status: {response.status}")
    print(f"   Added: {response.documents_added}")
    print(f"   Chunks: {response.chunks_created}")
    
    if response.status == "success" and response.documents_added > 0:
        print("   [SUCCESS] Incremental ingestion worked.")
    else:
        print(f"   [FAIL] Ingestion failed: {response.error if hasattr(response, 'error') else 'Unknown'}")

    # Test Query to ensure DB is accessible
    print("\n2. Testing Query after Ingestion")
    query_resp = cortex_ops.query("What is Mnemonic Caching?", max_results=1)
    if query_resp.status == "success":
         print(f"   [SUCCESS] Query worked. Found {len(query_resp.results)} results.")
    else:
         print(f"   [FAIL] Query failed: {query_resp.error}")

if __name__ == "__main__":
    test_ingestion()
