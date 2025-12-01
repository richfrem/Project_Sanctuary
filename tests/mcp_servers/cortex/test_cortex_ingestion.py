import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from mcp_servers.cognitive.cortex.operations import CortexOperations
from langchain_core.documents import Document

@pytest.fixture
def mock_cortex_deps():
    """Mock dependencies for CortexOperations ingestion."""
    with patch("mcp_servers.cognitive.cortex.operations.Chroma") as mock_chroma, \
         patch("mcp_servers.cognitive.cortex.operations.LocalFileStore") as mock_lfs, \
         patch("mcp_servers.cognitive.cortex.operations.create_kv_docstore") as mock_kv, \
         patch("mcp_servers.cognitive.cortex.operations.ParentDocumentRetriever") as mock_pdr, \
         patch("mcp_servers.cognitive.cortex.operations.NomicEmbeddings") as mock_nomic, \
         patch("mcp_servers.cognitive.cortex.operations.DirectoryLoader") as mock_dir_loader, \
         patch("mcp_servers.cognitive.cortex.operations.TextLoader") as mock_text_loader, \
         patch("mcp_servers.cognitive.cortex.operations.RecursiveCharacterTextSplitter") as mock_splitter:
        
        # Mock splitter to return predictable chunks
        mock_splitter_instance = mock_splitter.return_value
        mock_splitter_instance.split_documents.return_value = [
            Document(page_content="chunk1"),
            Document(page_content="chunk2")
        ]
        
        yield {
            "chroma": mock_chroma,
            "lfs": mock_lfs,
            "kv": mock_kv,
            "pdr": mock_pdr,
            "nomic": mock_nomic,
            "dir_loader": mock_dir_loader,
            "text_loader": mock_text_loader,
            "splitter": mock_splitter
        }

def test_initialization(temp_project_root):
    """Test CortexOperations initialization."""
    ops = CortexOperations(str(temp_project_root))
    assert ops.project_root == temp_project_root

def test_ingest_full(mock_cortex_deps, temp_project_root):
    """Test full ingestion flow with accurate chunk counting."""
    ops = CortexOperations(str(temp_project_root))
    
    # Mock DirectoryLoader to return documents
    mock_loader_instance = mock_cortex_deps["dir_loader"].return_value
    mock_loader_instance.load.return_value = [
        Document(page_content="Test content 1", metadata={"source": "doc1.md"}),
        Document(page_content="Test content 2", metadata={"source": "doc2.md"})
    ]
    
    # Create a dummy source directory
    (temp_project_root / "00_CHRONICLE").mkdir(exist_ok=True)
    
    result = ops.ingest_full(purge_existing=False, source_directories=["00_CHRONICLE"])
    
    assert result.status == "success"
    assert result.documents_processed == 2
    assert result.chunks_created == 4  # 2 docs * 2 chunks each (from mock splitter)
    
    # Verify add_documents was called
    mock_pdr_instance = mock_cortex_deps["pdr"].return_value
    mock_pdr_instance.add_documents.assert_called()

def test_ingest_incremental(mock_cortex_deps, temp_project_root):
    """Test incremental ingestion flow with accurate chunk counting."""
    ops = CortexOperations(str(temp_project_root))
    
    # Create a dummy file
    dummy_file = temp_project_root / "test_doc.md"
    dummy_file.write_text("Test content")
    
    # Mock TextLoader
    mock_loader_instance = mock_cortex_deps["text_loader"].return_value
    mock_loader_instance.load.return_value = [
        Document(page_content="Test content", metadata={"source": str(dummy_file)})
    ]
    
    result = ops.ingest_incremental(file_paths=[str(dummy_file)])
    
    assert result.status == "success"
    assert result.documents_added == 1
    assert result.chunks_created == 2  # 1 doc * 2 chunks (from mock splitter)
    
    # Verify add_documents was called
    mock_pdr_instance = mock_cortex_deps["pdr"].return_value
    mock_pdr_instance.add_documents.assert_called()

def test_ingest_incremental_invalid_file(mock_cortex_deps, temp_project_root):
    """Test incremental ingestion with invalid file."""
    ops = CortexOperations(str(temp_project_root))
    
    result = ops.ingest_incremental(file_paths=["/non/existent/file.md"])
    
    assert result.documents_added == 0
    assert result.error == "No valid files to ingest"

def test_chunks_created_accuracy(mock_cortex_deps, temp_project_root):
    """Test that chunks_created is accurately calculated, not hardcoded to 0."""
    ops = CortexOperations(str(temp_project_root))
    
    # Mock DirectoryLoader
    mock_loader_instance = mock_cortex_deps["dir_loader"].return_value
    mock_loader_instance.load.return_value = [
        Document(page_content="Test content", metadata={"source": "doc.md"})
    ]
    
    (temp_project_root / "00_CHRONICLE").mkdir(exist_ok=True)
    
    result = ops.ingest_full(purge_existing=False, source_directories=["00_CHRONICLE"])
    
    # Critical assertion: chunks_created should NOT be 0
    assert result.chunks_created > 0, "Bug: chunks_created should not be hardcoded to 0"
    assert result.chunks_created == 2  # 1 doc * 2 chunks (from mock splitter)
