#!/usr/bin/env python3
import chromadb
import sys
from pathlib import Path

# Add project root to sys.path for internal imports
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from mcp_servers.lib.utils.env_helper import get_env_variable, load_env
from langchain_nomic import NomicEmbeddings

# ============================================================================
# Configuration (from Environment)
# ============================================================================
load_env()

CHROMA_HOST = get_env_variable("CHROMA_HOST", required=False) or "127.0.0.1"
CHROMA_PORT = int(get_env_variable("CHROMA_PORT", required=False) or 8110)

# Expected collections
CHILD_COLLECTION = get_env_variable("CHROMA_CHILD_COLLECTION", required=False) or "child_chunks_v5"
PARENT_STORE = get_env_variable("CHROMA_PARENT_STORE", required=False) or "parent_documents_v5"

def test_embeddings():
    print("\n=== Nomic (Local) Embedding Check ===")
    try:
        print("Initializing NomicEmbeddings in local mode...")
        embeddings = NomicEmbeddings(
            model="nomic-embed-text-v1.5",
            inference_mode="local"
        )
        # Test query
        test_text = "Project Sanctuary initialization"
        print(f"Testing embedding for: '{test_text}'")
        vector = embeddings.embed_query(test_text)
        print(f"Status: SUCCESS")
        print(f"Vector dimensions: {len(vector)}")
    except Exception as e:
        print(f"[ERROR] Nomic embedding failed: {e}")
        print("Tip: Ensure the 'nomic' pip package is installed and you have the model weights locally.")

def test_chroma():
    print("\n=== ChromaDB Status Check ===")
    print(f"Connecting to: http://{CHROMA_HOST}:{CHROMA_PORT}")
    
    try:
        # We use HttpClient which doesn't require an embedding function for metadata/peek operations
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        
        # 1. List all collections
        collections = client.list_collections()
        col_names = [c.name for c in collections]
        print(f"\nAvailable Collections ({len(col_names)}):")
        for name in col_names:
            print(f"  - {name}")
            
        # 2. Check specific collections requested by user
        for col_name in [CHILD_COLLECTION, PARENT_STORE]:
            print(f"\n--- Investigating: {col_name} ---")
            if col_name in col_names:
                collection = client.get_collection(col_name)
                count = collection.count()
                print(f"Status: FOUND")
                print(f"Count:  {count} items")
                
                if count > 0:
                    print("Sample Data (Peek):")
                    peek = collection.peek(limit=1)
                    # Show ID and Metadata keys to avoid cluttering output
                    if peek['ids']:
                        print(f"  ID: {peek['ids'][0]}")
                        if peek['metadatas']:
                            print(f"  Metadata Keys: {list(peek['metadatas'][0].keys())}")
                        if peek['documents'] and peek['documents'][0]:
                            doc_snippet = peek['documents'][0][:150].replace('\n', ' ')
                            print(f"  Content Snippet: {doc_snippet}...")
                    else:
                        print("  (Collection is empty despite positive count?)")
            else:
                print(f"Status: NOT FOUND in Chroma collections.")
                if col_name == PARENT_STORE:
                    # Check if it exists as a FileStore on disk
                    file_store_path = project_root / CHROMA_DATA_PATH / col_name
                    print(f"Checking FileStore path: {file_store_path}")
                    if file_store_path.exists() and file_store_path.is_dir():
                        count = sum(1 for _ in file_store_path.glob("*.json"))
                        print(f"Status: FOUND (as FileStore on Disk)")
                        print(f"Count:  {count} parent documents")
                        if count > 0:
                            sample_file = next(file_store_path.glob("*.json"))
                            print(f"Sample File: {sample_file.name}")
                    else:
                        print(f"Status: NOT FOUND on disk either.")

    except Exception as e:
        print(f"\n[ERROR] Could not connect to ChromaDB: {e}")
        print("Make sure your Podman/Docker container is running and port 8110 is mapped.")

if __name__ == "__main__":
    # Add CHROMA_DATA_PATH to global for the FileStore check
    CHROMA_DATA_PATH = get_env_variable("CHROMA_DATA_PATH", required=False) or ".vector_data"
    test_embeddings()
    test_chroma()
