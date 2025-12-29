import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from mcp_servers.rag_cortex.operations import CortexOperations

def test_incremental():
    print(f"Testing ingest_incremental on {project_root}")
    ops = CortexOperations(project_root)
    
    test_file = project_root / "testing_rag_refactor.py"
    if not test_file.exists():
        print("❌ Test file missing!")
        return

    print(f"Ingesting {test_file}...")
    try:
        res = ops.ingest_incremental(file_paths=[str(test_file)])
        print(f"Result: {res.status}")
        print(f"Docs Added: {res.documents_added}")
        print(f"Chunks: {res.chunks_created}")
        
        if res.status == "success" and res.documents_added > 0:
            print("✅ Incremental ingest successful")
        else:
            print(f"❌ Ingest failed or no docs added: {res.error if hasattr(res, 'error') else 'Unknown'}")
            
    except Exception as e:
        print(f"❌ Exception during ingest: {e}")

if __name__ == "__main__":
    test_incremental()
