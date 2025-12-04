"""
RAG Cortex Setup Validation Script

This script validates the complete RAG Cortex setup:
1. Verifies ChromaDB container is running
2. Tests database connectivity
3. Checks if data exists
4. Performs full ingestion if needed
5. Runs sample queries to validate content
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import chromadb
from mcp_servers.rag_cortex.operations import CortexOperations
from mcp_servers.rag_cortex.container_manager import ensure_chromadb_running


def test_chromadb_connection():
    """Test 1: Verify ChromaDB is accessible"""
    print("\n" + "="*60)
    print("TEST 1: ChromaDB Connection")
    print("="*60)
    
    try:
        chroma_host = os.getenv("CHROMA_HOST", "localhost")
        chroma_port = int(os.getenv("CHROMA_PORT", "8000"))
        
        print(f"Connecting to ChromaDB at {chroma_host}:{chroma_port}...")
        client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        
        collections = client.list_collections()
        print(f"✓ Connected successfully!")
        print(f"✓ Found {len(collections)} collections")
        
        return True, client
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False, None


def test_data_exists(client):
    """Test 2: Check if data exists in ChromaDB"""
    print("\n" + "="*60)
    print("TEST 2: Data Existence Check")
    print("="*60)
    
    try:
        child_collection_name = os.getenv("CHROMA_CHILD_COLLECTION", "child_chunks_v5")
        
        collections = client.list_collections()
        collection_names = [c.name for c in collections]
        
        if child_collection_name in collection_names:
            collection = client.get_collection(name=child_collection_name)
            count = collection.count()
            print(f"✓ Collection '{child_collection_name}' exists")
            print(f"✓ Contains {count} chunks")
            return count > 0, count
        else:
            print(f"✗ Collection '{child_collection_name}' not found")
            return False, 0
            
    except Exception as e:
        print(f"✗ Error checking data: {e}")
        return False, 0


def perform_full_ingestion():
    """Test 3: Perform full data ingestion"""
    print("\n" + "="*60)
    print("TEST 3: Full Data Ingestion")
    print("="*60)
    
    try:
        ops = CortexOperations(str(project_root))
        
        print("Starting full ingestion (this may take a few minutes)...")
        result = ops.ingest_full(purge_existing=True)
        
        if result.status == "success":
            print(f"✓ Ingestion successful!")
            print(f"  - Documents processed: {result.documents_processed}")
            print(f"  - Chunks created: {result.chunks_created}")
            print(f"  - Time: {result.ingestion_time_ms/1000:.2f}s")
            return True, result.chunks_created
        else:
            print(f"✗ Ingestion failed: {result.error}")
            return False, 0
            
    except Exception as e:
        print(f"✗ Ingestion error: {e}")
        import traceback
        traceback.print_exc()
        return False, 0


def test_sample_query():
    """Test 4: Run sample query to validate content"""
    print("\n" + "="*60)
    print("TEST 4: Sample Query Validation")
    print("="*60)
    
    try:
        ops = CortexOperations(str(project_root))
        
        test_query = "What is the MCP architecture?"
        print(f"Query: '{test_query}'")
        
        result = ops.query(query=test_query, max_results=3)
        
        if result.status == "success" and result.results:
            print(f"✓ Query successful!")
            print(f"  - Found {len(result.results)} results")
            print(f"\nTop result preview:")
            top_result = result.results[0]
            print(f"  Source: {top_result.metadata.get('source', 'Unknown')}")
            print(f"  Content: {top_result.content[:200]}...")
            return True
        else:
            print(f"✗ Query failed or returned no results")
            if result.error:
                print(f"  Error: {result.error}")
            return False
            
    except Exception as e:
        print(f"✗ Query error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run complete setup validation"""
    print("\n" + "🚀 RAG CORTEX SETUP VALIDATION" + "\n")
    
    # Ensure ChromaDB container is running
    print("Checking ChromaDB container...")
    success, message = ensure_chromadb_running(str(project_root))
    if success:
        print(f"✓ {message}")
    else:
        print(f"✗ {message}")
        print("\nPlease ensure Podman is running and try again.")
        return False
    
    # Test 1: Connection
    connected, client = test_chromadb_connection()
    if not connected:
        print("\n❌ Setup validation FAILED: Cannot connect to ChromaDB")
        return False
    
    # Test 2: Data existence
    has_data, chunk_count = test_data_exists(client)
    
    # Test 3: Ingestion (if needed)
    if not has_data:
        print("\nNo data found. Performing initial ingestion...")
        ingested, new_chunks = perform_full_ingestion()
        if not ingested:
            print("\n❌ Setup validation FAILED: Ingestion failed")
            return False
    else:
        print(f"\n✓ Data already exists ({chunk_count} chunks)")
    
    # Test 4: Query validation
    query_ok = test_sample_query()
    if not query_ok:
        print("\n❌ Setup validation FAILED: Query test failed")
        return False
    
    # All tests passed
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED - RAG Cortex is ready!")
    print("="*60)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
