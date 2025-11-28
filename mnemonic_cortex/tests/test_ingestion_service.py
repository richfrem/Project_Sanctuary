import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from mnemonic_cortex.app.services.ingestion_service import IngestionService
from langchain_core.documents import Document

@pytest.fixture
def mock_ingestion_deps():
    """Mock dependencies for IngestionService."""
    with patch("mnemonic_cortex.app.services.ingestion_service.Chroma") as mock_chroma, \
         patch("mnemonic_cortex.app.services.ingestion_service.LocalFileStore") as mock_lfs, \
         patch("mnemonic_cortex.app.services.ingestion_service.EncoderBackedStore") as mock_ebs, \
         patch("mnemonic_cortex.app.services.ingestion_service.ParentDocumentRetriever") as mock_pdr, \
         patch("mnemonic_cortex.app.services.ingestion_service.NomicEmbeddings") as mock_nomic, \
         patch("mnemonic_cortex.app.services.ingestion_service.DirectoryLoader") as mock_dir_loader, \
         patch("mnemonic_cortex.app.services.ingestion_service.TextLoader") as mock_text_loader:
        
        yield {
            "chroma": mock_chroma,
            "lfs": mock_lfs,
            "ebs": mock_ebs,
            "pdr": mock_pdr,
            "nomic": mock_nomic,
            "dir_loader": mock_dir_loader,
            "text_loader": mock_text_loader
        }

def test_initialization(temp_project_root):
    """Test service initialization."""
    service = IngestionService(str(temp_project_root))
    assert service.project_root == temp_project_root
    assert service.db_path == "chroma_db"

def test_ingest_full(mock_ingestion_deps, temp_project_root):
    """Test full ingestion flow."""
    service = IngestionService(str(temp_project_root))
    
    # Mock DirectoryLoader to return documents
    mock_loader_instance = mock_ingestion_deps["dir_loader"].return_value
    mock_loader_instance.load.return_value = [
        Document(page_content="Test content 1", metadata={"source": "doc1.md"}),
        Document(page_content="Test content 2", metadata={"source": "doc2.md"})
    ]
    
    # Create a dummy source directory
    (temp_project_root / "00_CHRONICLE").mkdir(exist_ok=True)
    
    result = service.ingest_full(purge_existing=False, source_directories=["00_CHRONICLE"])
    
    assert result["status"] == "success"
    assert result["documents_processed"] == 2
    
    # Verify add_documents was called
    mock_pdr_instance = mock_ingestion_deps["pdr"].return_value
    mock_pdr_instance.add_documents.assert_called()

def test_ingest_incremental(mock_ingestion_deps, temp_project_root):
    """Test incremental ingestion flow."""
    service = IngestionService(str(temp_project_root))
    
    # Create a dummy file
    dummy_file = temp_project_root / "test_doc.md"
    dummy_file.write_text("Test content")
    
    # Mock TextLoader
    mock_loader_instance = mock_ingestion_deps["text_loader"].return_value
    mock_loader_instance.load.return_value = [
        Document(page_content="Test content", metadata={"source": str(dummy_file)})
    ]
    
    result = service.ingest_incremental(file_paths=[str(dummy_file)])
    
    assert result["status"] == "success"
    assert result["added"] == 1
    
    # Verify add_documents was called
    mock_pdr_instance = mock_ingestion_deps["pdr"].return_value
    mock_pdr_instance.add_documents.assert_called()

def test_ingest_incremental_invalid_file(mock_ingestion_deps, temp_project_root):
    """Test incremental ingestion with invalid file."""
    service = IngestionService(str(temp_project_root))
    
    result = service.ingest_incremental(file_paths=["/non/existent/file.md"])
    
    assert result["added"] == 0
    assert result["error"] == "No valid files to ingest"
