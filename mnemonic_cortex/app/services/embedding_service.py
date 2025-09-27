"""
Embedding Service (app/services/embedding_service.py)

This service provides a singleton wrapper for the Nomic embedding model used throughout the Mnemonic Cortex RAG system.
It ensures efficient resource management by maintaining a single instance of the embedding model.

Role in RAG Pipeline:
- Converts text (documents during ingestion, queries during retrieval) into high-dimensional vectors.
- Used in both Ingestion Pipeline (to embed document chunks) and Query Pipeline (to embed user questions).
- Enables semantic similarity searches by providing consistent vector representations.

Dependencies:
- Nomic Embeddings: An open-source, local-first embedding model (nomic-embed-text-v1.5).
- Runs in local inference mode; no external API calls or cloud dependencies.
- LangChain integration via langchain_community.embeddings.NomicEmbeddings.

Note: Implemented as a singleton to avoid redundant model loading and memory usage.
"""

from langchain_nomic import NomicEmbeddings

class EmbeddingService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            print("[EmbeddingService] Creating new instance...")
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance.model = NomicEmbeddings(
                model="nomic-embed-text-v1.5",
                inference_mode="local"
            )
            print("[EmbeddingService] Nomic embedding model initialized.")
        return cls._instance

    def get_embedding_model(self):
        """Returns the initialized embedding model."""
        return self.model