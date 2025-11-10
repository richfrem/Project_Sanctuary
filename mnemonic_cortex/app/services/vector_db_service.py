"""
Vector Database Service (app/services/vector_db_service.py)

This service manages the interaction with the ChromaDB vector database in the Mnemonic Cortex RAG system.
It provides a clean interface for loading the persisted Parent Document Retriever stores and creating
retrievers that return full parent documents based on relevant child chunks.

Role in RAG Pipeline:
- Loads the pre-built Parent Document Retriever stores from disk (populated by the ingestion script).
- Configures a ParentDocumentRetriever for performing similarity searches that return full documents.
- Serves as the retrieval component in the Query Pipeline, returning complete context documents for user queries.

Phase 1 Implementation (Parent Document Retriever):
- Stores document chunks in ChromaDB collection for precise semantic retrieval
- Stores full parent documents in separate ChromaDB collection for complete context
- Custom retriever finds relevant chunks, then returns corresponding full parent documents
- Prevents context fragmentation by ensuring complete document access
- Persistent storage using ChromaDB for both chunks and parent documents

Dependencies:
- ChromaDB: The local vector database must be initialized and populated via scripts/ingest.py.
- EmbeddingService: Provides the Nomic embedding model used for vectorizing queries and stored documents.
- Environment: Relies on DB_PATH from .env to locate the database directory.

Note: ChromaDB runs locally as a file-based database; no external server required.
"""

import os
from langchain_chroma import Chroma
from langchain_core.documents import Document
from typing import List
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from mnemonic_cortex.app.services.embedding_service import EmbeddingService
class ParentDocumentRetrieverCustom:
    """Custom Parent Document Retriever that uses two ChromaDB collections."""

    def __init__(self, chunks_vectorstore: Chroma, parents_vectorstore: Chroma):
        self.chunks_vectorstore = chunks_vectorstore
        self.parents_vectorstore = parents_vectorstore

    def invoke(self, query: str) -> List[Document]:
        """Find relevant chunks, then return corresponding parent documents."""
        # Find relevant chunks
        chunk_results = self.chunks_vectorstore.similarity_search(query, k=5)

        if not chunk_results:
            return []

        # Get unique source files from chunks
        source_files = set()
        for chunk in chunk_results:
            source_file = chunk.metadata.get('source_file')
            if source_file:
                source_files.add(source_file)

        # Retrieve full parent documents for these source files
        parent_docs = []
        for source_file in source_files:
            # Search parents vectorstore for documents with matching source_file
            # Use metadata filter to find exact matches
            filter_dict = {"source_file": source_file}
            parent_results = self.parents_vectorstore.get(where=filter_dict, limit=1)
            if parent_results['documents']:
                # Reconstruct document with metadata
                doc = Document(
                    page_content=parent_results['documents'][0],
                    metadata=parent_results['metadatas'][0] if parent_results['metadatas'] else {}
                )
                parent_docs.append(doc)

        return parent_docs[:5]  # Return top 5 parent documents

    def get_relevant_documents(self, query: str) -> List[Document]:
        """Find relevant chunks, then return corresponding parent documents."""
        # Find relevant chunks
        chunk_results = self.chunks_vectorstore.similarity_search(query, k=5)

        if not chunk_results:
            return []

        # Get unique source files from chunks
        source_files = set()
        for chunk in chunk_results:
            source_file = chunk.metadata.get('source_file')
            if source_file:
                source_files.add(source_file)

        # Retrieve full parent documents for these source files
        parent_docs = []
        for source_file in source_files:
            # Search parents vectorstore for documents with matching source_file
            parent_results = self.parents_vectorstore.similarity_search(
                f"source_file:{source_file}", k=1, filter={"source_file": source_file}
            )
            parent_docs.extend(parent_results)

        return parent_docs[:5]  # Return top 5 parent documents

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version of get_relevant_documents."""
        return self.get_relevant_documents(query)

class VectorDBService:
    def __init__(self) -> None:
        """
        Initializes the VectorDBService, loading the persistent ChromaDB with Parent Document Retriever.
        """
        print("[VectorDBService] Initializing with Parent Document Retriever...")
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        db_path = os.getenv("DB_PATH")
        full_db_path = os.path.join(project_root, 'mnemonic_cortex', db_path)

        if not os.path.exists(full_db_path):
            raise FileNotFoundError(f"ChromaDB not found at '{full_db_path}'. Please run the ingestion script first.")

        embedding_service = EmbeddingService()
        embedding_model = embedding_service.get_embedding_model()

        # Phase 1: Load dual vector stores for Parent Document architecture
        chunks_store_path = os.path.join(full_db_path, "chunks")
        parents_store_path = os.path.join(full_db_path, "parents")

        if not os.path.exists(chunks_store_path) or not os.path.exists(parents_store_path):
            raise FileNotFoundError(f"Parent Document stores not found. Please re-run the ingestion script.")

        # Load chunks vectorstore
        chunks_vectorstore = Chroma(
            collection_name="document_chunks",
            embedding_function=embedding_model,
            persist_directory=chunks_store_path
        )

        # Load parents vectorstore
        parents_vectorstore = Chroma(
            collection_name="parent_documents",
            embedding_function=embedding_model,
            persist_directory=parents_store_path
        )

        # Create custom Parent Document Retriever
        self.retriever = ParentDocumentRetrieverCustom(chunks_vectorstore, parents_vectorstore)

        print(f"[VectorDBService] Successfully loaded Parent Document Retriever from '{full_db_path}'.")

    def get_retriever(self):
        """Returns the configured retriever for similarity searches."""
        return self.retriever