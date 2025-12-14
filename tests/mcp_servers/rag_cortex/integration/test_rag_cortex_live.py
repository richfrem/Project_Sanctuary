import pytest
import os
import time
import chromadb
from langchain_chroma import Chroma
from pathlib import Path
from tests.mcp_servers.base.base_integration_test import BaseIntegrationTest
from mcp_servers.rag_cortex.operations import CortexOperations
from mcp_servers.rag_cortex.validator import CortexValidator

class TestRAGCortexLive(BaseIntegrationTest):
    """
    Layer 2 Integration Test for RAG Cortex.
    Validates REAL connectivity to ChromaDB and Ollama.
    """
    
    def get_required_services(self):
        return [
            ("localhost", 8000, "ChromaDB"),
            ("localhost", 11434, "Ollama")
        ]

    @pytest.fixture
    def cortex_ops(self, tmp_path):
        """Initialize real CortexOperations connected to real ChromaDB."""
        # Use a temporary directory for file storage to avoid polluting real data
        project_root = tmp_path / "project_root"
        project_root.mkdir()
        (project_root / ".env").write_text("CHROMA_HOST=localhost\nCHROMA_PORT=8000\n")
        
        # Connect to REAL ChromaDB (local server)
        host = os.getenv("CHROMA_HOST", "localhost")
        port = int(os.getenv("CHROMA_PORT", "8000"))
        client = chromadb.HttpClient(host=host, port=port)
        
        ops = CortexOperations(str(project_root), client=client)
        
        # Override collections to test-specific ones to avoid wrecking production data
        ops.child_collection_name = f"test_child_{int(time.time())}"
        ops.parent_collection_name = f"test_parent_{int(time.time())}"
        
        # Re-init vectorstore with new collection name
        ops.vectorstore = Chroma(
            client=client,
            collection_name=ops.child_collection_name,
            embedding_function=ops.embedding_model
        )
        
        return ops

    def test_chroma_connectivity(self, cortex_ops):
        """Validate we can talk to ChromaDB."""
        heartbeat = cortex_ops.chroma_client.heartbeat()
        assert heartbeat is not None

    def test_ollama_embedding_generation(self, cortex_ops):
        """Validate Ollama is generating real embeddings."""
        text = "The quick brown fox jumps over the lazy dog."
        embedding = cortex_ops.embedding_model.embed_query(text)
        assert len(embedding) > 0
        assert isinstance(embedding, list)
        assert isinstance(embedding[0], float)

    def test_full_rag_cycle(self, cortex_ops):
        """
        Validate full Ingest -> Store -> Retrieve cycle with REAL components.
        """
        # 1. Create content
        source_dir = cortex_ops.project_root / "00_CHRONICLE"
        source_dir.mkdir()
        (source_dir / "test_doc.md").write_text(
            "# Live Test Document\n\nThis is a live integration test for Protocol 101."
        )
        
        # 2. Ingest
        result = cortex_ops.ingest_full(purge_existing=True, source_directories=["00_CHRONICLE"])
        assert result.status == "success"
        assert result.documents_processed == 1
        
        # 3. Query
        q_result = cortex_ops.query("Protocol 101", max_results=1)
        assert q_result.status == "success"
        if q_result.results:
            assert "Live Test Document" in q_result.results[0].content
            
    def teardown_method(self, method):
        """Cleanup test collections."""
        # Note: We rely on unique collection names per test run to avoid conflicts
        # But we should try to delete them if possible.
        pass
