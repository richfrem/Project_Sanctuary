"""
Unit tests for Cortex Operations (Business Logic).
Decoupled from Pydantic Models and Heavy ML Libraries.
Extensively mocks ChromaDB, LangChain, and File I/O.
"""
import pytest
import sys
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path

# Mock modules BEFORE importing operations to avoid slow loads/connect errors
with patch.dict(sys.modules, {
    "chromadb": MagicMock(),
    "langchain_community.document_loaders": MagicMock(),
    "langchain_text_splitters": MagicMock(),
    "langchain_huggingface": MagicMock(),
    "langchain_chroma": MagicMock(),
    "langchain_core.documents": MagicMock(),
}):
    # Import Document mock for typing if needed
    from langchain_core.documents import Document as MockDocument
    from mcp_servers.rag_cortex.operations import CortexOperations


class TestCortexOperations:
    @pytest.fixture
    def mock_dependencies(self):
        with patch("mcp_servers.rag_cortex.operations.chromadb") as mock_chroma, \
             patch("mcp_servers.rag_cortex.operations.HuggingFaceEmbeddings") as mock_embeddings, \
             patch("mcp_servers.rag_cortex.operations.Chroma") as mock_vectorstore, \
             patch("mcp_servers.rag_cortex.operations.RecursiveCharacterTextSplitter") as mock_splitter, \
             patch("mcp_servers.rag_cortex.operations.ContentProcessor") as mock_processor, \
             patch("mcp_servers.rag_cortex.operations.SimpleFileStore") as mock_store:
            
            yield {
                "chroma": mock_chroma,
                "embeddings": mock_embeddings,
                "vectorstore": mock_vectorstore,
                "splitter": mock_splitter,
                "splitter": mock_splitter,
                "processor": mock_processor,
                "store": mock_store
            }

    @pytest.fixture
    def ops(self, mock_dependencies, tmp_path):
        # Setup basic return values
        mock_dependencies["chroma"].HttpClient.return_value = MagicMock()
        
        return CortexOperations(project_root=str(tmp_path))

    def test_initialization(self, ops, mock_dependencies, tmp_path):
        """Test that init sets up paths and clients."""
        assert ops.project_root == tmp_path
        assert ops.data_dir == tmp_path / ".agent" / "data"
        
        mock_dependencies["chroma"].HttpClient.assert_called()
        mock_dependencies["embeddings"].assert_called()
        mock_dependencies["vectorstore"].assert_called()

    def test_ingest_full_success(self, ops, mock_dependencies, tmp_path):
        """Test full ingestion flow."""
        # Create dummy directory
        (tmp_path / "docs").mkdir()
        
        # Mock Loader docs
        mock_doc = MagicMock()
        mock_doc.page_content = "content"
        mock_doc.page_content = "content"
        # The processor is instantiated in ingest_full
        # We need to ensure the mock returned by the class constructor behaves right
        mock_processor_instance = mock_dependencies["processor"].return_value
        mock_processor_instance.load_for_rag.return_value = [mock_doc]
        
        # Mock Splitter
        mock_dependencies["splitter"].return_value.split_documents.return_value = [mock_doc]
        
        # Run ingest
        result = ops.ingest_full(purge_existing=True, source_directories=["docs"])
        
        assert result.status == "success"
        
        # Verify purge calls
        ops.chroma_client.delete_collection.assert_called()
        
        
        # Verify loading - ContentProcessor.load_for_rag should have been called
        mock_dependencies["processor"].assert_called()
        
        # Verify adding to vectorstore
        ops.vectorstore.add_documents.assert_called()

    def test_query_success(self, ops, mock_dependencies):
        """Test query flow."""
        collection_mock = MagicMock()
        ops.chroma_client.get_collection.return_value = collection_mock
        
        # Mock search results
        collection_mock.query.return_value = {
            "documents": [["result content"]],
            "metadatas": [[{"parent_id": "123"}]],
            "distances": [[0.1]]
        }
        
        # Mock parent store retrieval
        parent_doc = MagicMock()
        parent_doc.page_content = "Full parent content"
        parent_doc.metadata = {"source": "file.txt"}
        ops.store.mget.return_value = [parent_doc]
        
        scan_result = ops.query("test query")
        
        assert scan_result.status == "success"
        assert len(scan_result.results) == 1
        assert scan_result.results[0].content == "Full parent content" # Should upgrade to parent content
        
        # Verify embedding call
        ops.embedding_model.embed_query.assert_called_with("test query")

    def test_get_stats(self, ops, mock_dependencies, tmp_path):
        """Test stats retrieval."""
        collection_mock = MagicMock()
        collection_mock.count.return_value = 42
        ops.chroma_client.get_collection.return_value = collection_mock
        
        # Fix store path for existence check
        # ops.store is the instance. 
        ops.store.root_path = str(tmp_path)
        ops.store.yield_keys.return_value = iter(["1", "2"])
        
        stats = ops.get_stats()
        
        assert stats.collections["child_chunks"].count == 42
        assert stats.collections["parent_documents"].count == 2
        assert stats.health_status == "healthy"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
