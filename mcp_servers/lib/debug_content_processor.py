
import sys
import shutil
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from mcp_servers.lib.content_processor import ContentProcessor

def debug_traversal():
    """Debug ContentProcessor traversal in a temp-like path."""
    # Create fake temp dir
    fake_tmp = PROJECT_ROOT / "debug_tmp_env"
    if fake_tmp.exists():
        shutil.rmtree(fake_tmp)
    fake_tmp.mkdir()
    
    # Create fake project root structure inside
    # In the test, cortex_ops sets project_root = tmp_path / "project_root"
    test_root = fake_tmp / "project_root"
    test_root.mkdir()
    
    # Create subdir and file
    (test_root / "00_CHRONICLE").mkdir()
    (test_root / "00_CHRONICLE" / "test.md").write_text("# Test Content")
    
    print(f"DEBUG: Created test root at {test_root}")
    
    # Init Processor with this root
    processor = ContentProcessor(str(test_root))
    print(f"DEBUG: Processor root: {processor.project_root}")
    
    # Try traversing
    target_dir = test_root / "00_CHRONICLE"
    print(f"DEBUG: Traversing {target_dir}")
    
    files = list(processor.traverse_directory(target_dir))
    print(f"DEBUG: Found files: {files}")
    
    # Try loading for RAG
    # In operations.py, paths_to_scan = [str(self.project_root / d) ...]
    # But in ingest_incremental logic: path = self.project_root / path (if relative)
    
    # Test valid absolute path input (which traverse_directory yields)
    try:
        docs = list(processor.load_for_rag([str(target_dir)]))
        print(f"DEBUG: Docs found via load_for_rag: {len(docs)}")
        for doc in docs:
            print(f"  - {doc.metadata}")
    except Exception as e:
        print(f"ERROR: {e}")
        
    # Clean up
    shutil.rmtree(fake_tmp)

if __name__ == "__main__":
    debug_traversal()
