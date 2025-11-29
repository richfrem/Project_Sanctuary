import pytest
from unittest.mock import MagicMock, patch
import os
from mnemonic_cortex.app.services.vector_db_service import VectorDBService

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton instance before and after each test."""
    VectorDBService._instance = None
    yield
    VectorDBService._instance = None

@pytest.fixture
def mock_dependencies():
    """Mock external dependencies."""
    with patch("mnemonic_cortex.app.services.vector_db_service.Chroma") as mock_chroma, \
         patch("mnemonic_cortex.app.services.vector_db_service.LocalFileStore") as mock_lfs, \
         patch("mnemonic_cortex.app.services.vector_db_service.EncoderBackedStore") as mock_ebs, \
         patch("mnemonic_cortex.app.services.vector_db_service.ParentDocumentRetriever") as mock_pdr, \
         patch("mnemonic_cortex.app.services.vector_db_service.EmbeddingService") as mock_es:
        
        yield {
            "chroma": mock_chroma,
            "lfs": mock_lfs,
            "ebs": mock_ebs,
            "pdr": mock_pdr,
            "es": mock_es
        }

def test_initialization(mock_dependencies, temp_project_root):
    """Test that VectorDBService initializes correctly."""
    # Ensure paths exist so validation passes
    # (temp_project_root fixture already creates them in conftest)
    
    # We need to patch CHROMA_ROOT in the module to point to our temp dir
    with patch("mnemonic_cortex.app.services.vector_db_service.CHROMA_ROOT", temp_project_root / "mnemonic_cortex" / "chroma_db"):
        # Also patch os.path.exists to return True (or rely on real files)
        # Since we created dirs in conftest, real existence check should pass if paths match
        
        # We need to ensure _detect_collections finds our temp collections
        # conftest created 'test_child' and 'test_parent' dirs?
        # Let's check conftest logic.
        # It created (tmp_path / "mnemonic_cortex" / "chroma_db")
        # But _detect_collections looks for child_chunks* and parent_documents*
        # conftest env set: CHROMA_CHILD_COLLECTION=test_child
        # So it should look for 'test_child'
        
        # Create the specific collection dirs in temp root
        (temp_project_root / "mnemonic_cortex" / "chroma_db" / "child_chunks_v5").mkdir()
        (temp_project_root / "mnemonic_cortex" / "chroma_db" / "parent_documents_v5").mkdir()

        service = VectorDBService()
        
        assert service.initialized is True
        mock_dependencies["es"].assert_called_once()
        mock_dependencies["pdr"].assert_called_once()

def test_query(mock_dependencies, temp_project_root):
    """Test the query method."""
    with patch("mnemonic_cortex.app.services.vector_db_service.CHROMA_ROOT", temp_project_root / "mnemonic_cortex" / "chroma_db"):
        # Setup dirs
        (temp_project_root / "mnemonic_cortex" / "chroma_db" / "child_chunks_v5").mkdir(exist_ok=True)
        (temp_project_root / "mnemonic_cortex" / "chroma_db" / "parent_documents_v5").mkdir(exist_ok=True)
        
        service = VectorDBService()
        
        # Setup mock return
        mock_retriever = mock_dependencies["pdr"].return_value
        mock_retriever.invoke.return_value = ["doc1", "doc2"]
        
        results = service.query("test query")
        
        assert len(results) == 2
        assert results == ["doc1", "doc2"]
        mock_retriever.invoke.assert_called_with("test query")

def test_singleton_behavior(mock_dependencies, temp_project_root):
    """Ensure it acts as a singleton."""
    with patch("mnemonic_cortex.app.services.vector_db_service.CHROMA_ROOT", temp_project_root / "mnemonic_cortex" / "chroma_db"):
        (temp_project_root / "mnemonic_cortex" / "chroma_db" / "child_chunks_v5").mkdir(exist_ok=True)
        (temp_project_root / "mnemonic_cortex" / "chroma_db" / "parent_documents_v5").mkdir(exist_ok=True)
        
        s1 = VectorDBService()
        s2 = VectorDBService()
        
        assert s1 is s2
