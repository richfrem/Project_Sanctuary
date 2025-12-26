#!/usr/bin/env python3
"""
Project Sanctuary - ChromaDB Inspector
Tests ChromaDB connectivity from localhost and/or container network.

Usage:
  python inspect_chroma.py                 # Test localhost only (default from host)
  python inspect_chroma.py --host all      # Test both (container will fail from host)
  python inspect_chroma.py --host container # Test container network only
"""
import sys
import argparse
from pathlib import Path

try:
    import chromadb
except ImportError:
    print("Error: 'chromadb' module not found. Install with: pip install chromadb")
    sys.exit(1)

# Add project root based on .git marker
current = Path(__file__).resolve().parent
while not (current / ".git").exists():
    if current == current.parent:
        raise RuntimeError("Could not find Project_Sanctuary root (no .git folder found)")
    current = current.parent
project_root = current
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from mcp_servers.lib.env_helper import get_env_variable, load_env

# ============================================================================
# Configuration (from Environment)
# ============================================================================
load_env()

# Get host/port from env
CHROMA_HOST_ENV = get_env_variable("CHROMA_HOST", required=False) or "127.0.0.1"
CHROMA_PORT = int(get_env_variable("CHROMA_PORT", required=False) or 8110)
CHROMA_DATA_PATH = get_env_variable("CHROMA_DATA_PATH", required=False) or ".vector_data"

# Container network host (for containers, not accessible from host machine)
CONTAINER_HOST = "sanctuary_vector_db"

# Expected collections
CHILD_COLLECTION = get_env_variable("CHROMA_CHILD_COLLECTION", required=False) or "child_chunks_v5"
PARENT_STORE = get_env_variable("CHROMA_PARENT_STORE", required=False) or "parent_documents_v5"


def print_header(title: str):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")


def check_connection(host: str, port: int, host_name: str):
    """Check if ChromaDB is reachable."""
    print(f"\n--- Testing {host_name}: http://{host}:{port} ---")
    
    try:
        client = chromadb.HttpClient(host=host, port=port)
        # Test connection by listing collections
        collections = client.list_collections()
        print(f"  Status: ✅ ONLINE ({len(collections)} collections)")
        return client
            
    except Exception as e:
        print(f"  Status: ❌ FAILED ({e})")
        if host_name == "container":
            print(f"  Note: Container hostnames only resolve from inside containers")
        return None


def inspect_collections(client: chromadb.HttpClient) -> dict:
    """Inspect collections in ChromaDB."""
    results = {"collections": [], "status": "success"}
    
    try:
        collections = client.list_collections()
        col_names = [c.name for c in collections]
        
        print(f"  Available Collections ({len(col_names)}):")
        for name in col_names:
            print(f"    - {name}")
        
        # Check specific collections
        for col_name in [CHILD_COLLECTION, PARENT_STORE]:
            print(f"\n  --- {col_name} ---")
            if col_name in col_names:
                collection = client.get_collection(col_name)
                count = collection.count()
                print(f"  Status: FOUND ({count} items)")
                results["collections"].append({"name": col_name, "count": count})
                
                if count > 0:
                    peek = collection.peek(limit=1)
                    if peek['ids']:
                        print(f"  Sample ID: {peek['ids'][0][:50]}...")
                        if peek['metadatas'] and peek['metadatas'][0]:
                            print(f"  Metadata Keys: {list(peek['metadatas'][0].keys())}")
            else:
                print(f"  Status: NOT FOUND")
                # Check FileStore fallback
                file_store_path = project_root / CHROMA_DATA_PATH / col_name
                if file_store_path.exists() and file_store_path.is_dir():
                    count = sum(1 for _ in file_store_path.glob("*.json"))
                    print(f"  FileStore: FOUND ({count} files on disk)")
                    results["collections"].append({"name": col_name, "count": count, "type": "filestore"})
                    
    except Exception as e:
        print(f"  [ERROR] {e}")
        results["status"] = "error"
    
    return results


def test_embeddings() -> bool:
    """Test HuggingFace embeddings (optional)."""
    print_header("HuggingFace Embeddings Check")
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        
        print("  Initializing HuggingFaceEmbeddings (nomic-embed-text-v1.5)...")
        embeddings = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            model_kwargs={'device': 'cpu', 'trust_remote_code': True},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        test_text = "Project Sanctuary initialization"
        vector = embeddings.embed_query(test_text)
        print(f"  Status: ✅ SUCCESS")
        print(f"  Vector dimensions: {len(vector)}")
        return True
        
    except ImportError:
        print("  Status: ⚠️ SKIPPED (langchain_huggingface not installed)")
        return True  # Not a failure, just not tested
    except Exception as e:
        print(f"  Status: ❌ FAILED ({e})")
        return False


def run_tests(host_filter: str = "localhost", skip_embeddings: bool = False) -> dict:
    """Run tests on specified hosts."""
    print("Project Sanctuary - ChromaDB Inspector")
    print(f"CHROMA_HOST from .env: {CHROMA_HOST_ENV}")
    print(f"CHROMA_PORT from .env: {CHROMA_PORT}")
    
    # Build host list based on filter
    hosts = {}
    if host_filter in ("all", "localhost"):
        hosts["localhost"] = CHROMA_HOST_ENV
    if host_filter in ("all", "container"):
        hosts["container"] = CONTAINER_HOST
    
    if not hosts:
        print(f"Unknown host: {host_filter}. Use 'localhost', 'container', or 'all'.")
        return {}
    
    results = {}
    
    # 1. Connectivity Check
    print_header("1. Connectivity Check")
    for name, host in hosts.items():
        client = check_connection(host, CHROMA_PORT, name)
        results[name] = {"connected": client is not None, "client": client}
    
    # 2. Collection Inspection (only for connected hosts)
    print_header("2. Collection Inspection")
    for name in hosts:
        if results[name]["connected"]:
            print(f"\n--- {name} ---")
            results[name]["collections"] = inspect_collections(results[name]["client"])
        else:
            print(f"\n--- Skipping {name} (not connected) ---")
            results[name]["collections"] = None
    
    # 3. Embeddings Test (optional)
    if not skip_embeddings:
        results["embeddings"] = test_embeddings()
    
    # Summary
    print_header("Summary")
    for name in hosts:
        status = "✅ PASS" if results[name].get("connected") else "❌ FAIL"
        print(f"  {name}: {status}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ChromaDB Inspector - Test connectivity")
    parser.add_argument("--host", default="localhost",
                        choices=["all", "localhost", "container"],
                        help="Which host to test (default: localhost)")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip HuggingFace embeddings test")
    args = parser.parse_args()
    
    results = run_tests(args.host, args.skip_embeddings)
    
    # Exit code: 0 if at least one host passes, 1 otherwise
    host_results = [r for k, r in results.items() if k not in ("embeddings",)]
    if any(r.get("connected") for r in host_results):
        sys.exit(0)
    else:
        sys.exit(1)
