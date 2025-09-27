"""
Vector Database Service (app/services/vector_db_service.py)

This service manages the interaction with the ChromaDB vector database in the Mnemonic Cortex RAG system.
It provides a clean interface for loading the persisted vector store and creating retrievers for similarity searches.

Role in RAG Pipeline:
- Loads the pre-built ChromaDB from disk (populated by the ingestion script).
- Configures a retriever for performing similarity searches on embedded documents.
- Serves as the retrieval component in the Query Pipeline, returning relevant context chunks for user queries.

Dependencies:
- ChromaDB: The local vector database must be initialized and populated via scripts/ingest.py.
- EmbeddingService: Provides the Nomic embedding model used for vectorizing queries and stored documents.
- Environment: Relies on DB_PATH from .env to locate the database directory.

Note: ChromaDB runs locally as a file-based database; no external server required.
"""

import os
from langchain_chroma import Chroma
from mnemonic_cortex.app.services.embedding_service import EmbeddingService

class VectorDBService:
    def __init__(self):
        """
        Initializes the VectorDBService, loading the persistent ChromaDB.
        """
        print("[VectorDBService] Initializing...")
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        db_path = os.getenv("DB_PATH")
        full_db_path = os.path.join(project_root, 'mnemonic_cortex', db_path)

        if not os.path.exists(full_db_path):
            raise FileNotFoundError(f"ChromaDB not found at '{full_db_path}'. Please run the ingestion script first.")

        embedding_service = EmbeddingService()
        self.db = Chroma(
            persist_directory=full_db_path,
            embedding_function=embedding_service.get_embedding_model()
        )
        self.retriever = self.db.as_retriever()
        print(f"[VectorDBService] Successfully loaded ChromaDB from '{full_db_path}'.")

    def get_retriever(self):
        """Returns the configured retriever for similarity searches."""
        return self.retriever