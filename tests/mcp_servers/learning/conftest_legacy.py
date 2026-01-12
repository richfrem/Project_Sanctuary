import pytest
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

@pytest.fixture
def temp_project_root(tmp_path):
    """Create a temporary project root structure."""
    # Create standard directories
    (tmp_path / "mnemonic_cortex" / "chroma_db").mkdir(parents=True)
    (tmp_path / "00_CHRONICLE").mkdir()
    (tmp_path / "01_PROTOCOLS").mkdir()
    
    # Create .env file
    env_file = tmp_path / ".env"
    env_file.write_text("DB_PATH=chroma_db\nCHROMA_CHILD_COLLECTION=test_child\nCHROMA_PARENT_STORE=test_parent")
    
    return tmp_path

@pytest.fixture
def mock_chroma_client():
    """Mock ChromaDB client and collections."""
    with patch("chromadb.PersistentClient") as mock_client:
        mock_collection = MagicMock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        yield mock_client

@pytest.fixture
def mock_embedding_model():
    """Mock embedding function."""
    with patch("mnemonic_cortex.app.services.vector_db_service.NomicEmbedder") as mock_embed:
        mock_instance = mock_embed.return_value
        # Mock encode to return a dummy vector
        mock_instance.encode.return_value = [0.1] * 768
        yield mock_instance
