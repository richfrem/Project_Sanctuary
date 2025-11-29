import pytest
from unittest.mock import patch
from mnemonic_cortex.app.services.embedding_service import EmbeddingService

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset the singleton instance before and after each test."""
    EmbeddingService._instance = None
    yield
    EmbeddingService._instance = None

def test_initialization():
    """Test that EmbeddingService initializes the model correctly."""
    with patch("mnemonic_cortex.app.services.embedding_service.NomicEmbeddings") as mock_nomic:
        service = EmbeddingService()
        
        mock_nomic.assert_called_once_with(
            model="nomic-embed-text-v1.5",
            inference_mode="local"
        )
        assert service.model == mock_nomic.return_value

def test_singleton_behavior():
    """Ensure it acts as a singleton."""
    with patch("mnemonic_cortex.app.services.embedding_service.NomicEmbeddings"):
        s1 = EmbeddingService()
        s2 = EmbeddingService()
        
        assert s1 is s2

def test_get_embedding_model():
    """Test the getter method."""
    with patch("mnemonic_cortex.app.services.embedding_service.NomicEmbeddings"):
        service = EmbeddingService()
        model = service.get_embedding_model()
        assert model == service.model
