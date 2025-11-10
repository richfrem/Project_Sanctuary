# council_orchestrator/orchestrator/memory/cortex.py
# Mnemonic cortex vector database functionality

import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path

class CortexManager:
    """Manages the Mnemonic Cortex vector database for knowledge retrieval."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        # Access mnemonic_cortex at project root level (parent of council_orchestrator)
        self.chroma_client = chromadb.PersistentClient(path=str(project_root.parent / "mnemonic_cortex/chroma_db"))
        self.cortex_collection = self.chroma_client.get_or_create_collection(
            name="sanctuary_cortex",
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )

    def query_cortex(self, query_text: str, n_results: int = 3) -> str:
        """Query the cortex for relevant knowledge."""
        try:
            results = self.cortex_collection.query(query_texts=[query_text], n_results=n_results)
            context = "CONTEXT_PROVIDED: Here are the top results from the Mnemonic Cortex for your query:\n\n"
            for doc in results['documents'][0]:
                context += f"---\n{doc}\n---\n"
            return context
        except Exception as e:
            error_message = f"CONTEXT_ERROR: Cortex query failed: {e}"
            print(f"[CORTEX] {error_message}")
            return error_message

    def ingest_document(self, document: str, metadata: dict = None) -> bool:
        """Ingest a document into the cortex."""
        try:
            doc_id = f"doc_{hash(document) % 1000000}"
            self.cortex_collection.add(
                documents=[document],
                ids=[doc_id],
                metadatas=[metadata or {}]
            )
            return True
        except Exception as e:
            print(f"[CORTEX] Failed to ingest document: {e}")
            return False