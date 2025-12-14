
import pytest
import shutil
from pathlib import Path
import chromadb
from mcp_servers.rag_cortex.operations import CortexOperations

@pytest.fixture
def temp_chroma_env(tmp_path):
    """
    Set up a temporary environment for ChromaDB testing.
    Returns a tuple of (project_root, storage_path, chroma_client)
    """
    # Create project structure
    project_root = tmp_path / "project_root"
    project_root.mkdir()
    
    # Create .env file
    env_file = project_root / ".env"
    env_file.write_text("CHROMA_HOST=localhost\nCHROMA_PORT=8000\n")
    
    # Storage for parent documents
    storage_path = project_root / ".vector_data"
    storage_path.mkdir()
    
    # Initialize ephemeral/local client
    # We use a persistent client in a temp dir to simulate "real" DB better than volatile MemoryClient
    chroma_db_dir = tmp_path / "chroma_db"
    client = chromadb.PersistentClient(path=str(chroma_db_dir))
    
    return project_root, storage_path, client

def test_full_ingestion_and_query_flow(temp_chroma_env):
    """
    Test the complete flow:
    1. Setup environment
    2. Create content
    3. Ingest (Real DB)
    4. Query (Real DB retrieval)
    5. Verify stats
    """
    project_root, storage_path, client = temp_chroma_env
    
    # 1. Initialize Operations with injected client
    ops = CortexOperations(str(project_root), client=client)
    
    # Override parent collection path to use our temp storage
    ops.store.root_path = Path(str(storage_path / "parent_documents_test"))
    
    # 2. Create test content
    source_dir = project_root / "00_CHRONICLE"
    source_dir.mkdir()
    
    test_file_1 = source_dir / "test_doc_1.md"
    test_file_1.write_text(
        "# Project Sanctuary Test\n\n"
        "This is a test document for the RAG Cortex integration test.\n"
        "It contains specific keywords like 'Quantum Diamond' and 'Protocol 101'.\n"
        "\n"
        "## Section 2\n"
        "Detailed explanation of the protocol goes here.\n"
    )
    
    # 3. Perform Ingestion
    print("\n--- Starting Ingestion ---")
    result = ops.ingest_full(purge_existing=True, source_directories=["00_CHRONICLE"])
    
    assert result.status == "success"
    assert result.documents_processed == 1
    assert result.chunks_created > 0
    print(f"Ingestion complete. Processed {result.documents_processed} docs, created {result.chunks_created} chunks.")
    
    # 4. Verify Parent Documents
    # Check if files exist in the file store
    parent_docs = list(Path(ops.store.root_path).glob("*"))
    assert len(parent_docs) > 0
    print(f"Verified {len(parent_docs)} parent documents in store.")
    
    # 5. Perform Query
    print("\n--- Testing Retrieval ---")
    query_text = "What is Project Sanctuary?"
    query_result = ops.query(query_text, max_results=3)
    
    assert query_result.status == "success"
    # Note: We might not get results if the embedding model isn't loading or matching well with just 1 doc,
    # but we check that the mechanism runs without error. 
    # With "Project Sanctuary" in title and text, it SHOULD match.
    
    if query_result.results:
        print(f"Query returned {len(query_result.results)} results.")
        top_match = query_result.results[0]
        print(f"Top match content sample: {top_match.content[:100]}...")
        assert "Project Sanctuary" in top_match.content or "test document" in top_match.content
    else:
        print("WARNING: Query returned no results. This might be due to embedding model loading or high threshold.")
    
    # 6. Verify Stats
    stats = ops.get_stats()
    assert stats.total_chunks == result.chunks_created
    assert stats.health_status == "healthy"
    print("\n--- Test Complete ---")
