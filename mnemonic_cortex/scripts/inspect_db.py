"""
Database Inspection Script (scripts/inspect_db.py)

This script provides a command-line interface for inspecting the contents of the Mnemonic Cortex ChromaDB vector database.
It allows users to verify the ingestion process by displaying document counts, metadata, and content previews.

Role in RAG Pipeline:
- Diagnostic tool for the Ingestion Pipeline.
- Enables verification that documents were properly chunked, embedded, and stored.
- Supports debugging and quality assurance of the vector database.

Dependencies:
- ChromaDB: The vector database to inspect.
- NomicEmbeddings: For loading the database with the correct embedding function.
- Environment configuration: Relies on .env for DB_PATH.
- Project structure: Uses find_project_root() for path resolution.

Usage:
    python mnemonic_cortex/scripts/inspect_db.py
"""

import os
import argparse
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_nomic import NomicEmbeddings

# This setup is similar to our other scripts to ensure paths are correct
def find_project_root() -> str:
    current_path = os.path.abspath(os.path.dirname(__file__))
    while True:
        if '.git' in os.listdir(current_path):
            return current_path
        parent_path = os.path.dirname(current_path)
        if parent_path == current_path:
            raise FileNotFoundError("Could not find project root (.git folder).")
        current_path = parent_path

def setup_environment(project_root: str) -> None:
    dotenv_path = os.path.join(project_root, 'mnemonic_cortex', '.env')
    load_dotenv(dotenv_path=dotenv_path)

def main() -> None:
    """
    A command-line tool to inspect the contents of the Mnemonic Cortex ChromaDB.
    """
    project_root = find_project_root()
    setup_environment(project_root)
    db_path = os.getenv("DB_PATH")
    full_db_path = os.path.join(project_root, 'mnemonic_cortex', db_path)

    if not os.path.exists(full_db_path):
        print(f"ERROR: Database not found at '{full_db_path}'. Please run the ingestion script first.")
        return

    print(f"--- Connecting to ChromaDB at '{full_db_path}' ---")
    embedding_model = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
    
    db = Chroma(
        persist_directory=full_db_path,
        embedding_function=embedding_model
    )

    # --- Inspection Functions ---

    # Get the total number of documents
    total_docs = db._collection.count()
    print(f"\nTotal documents in the database: {total_docs}")

    # Fetch a few documents to see what they look like
    print("\n--- Sample of Stored Documents (first 5) ---")
    retrieved_docs = db.get(limit=5, include=["metadatas", "documents"])
    
    for i in range(len(retrieved_docs["ids"])):
        print(f"\n--- Document {i+1} ---")
        print(f"ID: {retrieved_docs['ids'][i]}")
        print(f"Metadata: {retrieved_docs['metadatas'][i]}")
        # Print the first 150 characters of the document content
        print(f"Content Preview: {retrieved_docs['documents'][i][:150]}...")

if __name__ == "__main__":
    main()