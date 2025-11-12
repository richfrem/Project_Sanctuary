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
from pathlib import Path
from langchain_community.vectorstores import Chroma
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
    # Load the single repo-root .env (per project policy). Do not rely on per-subpackage .env files.
    dotenv_path = os.path.join(project_root, '.env')
    load_dotenv(dotenv_path=dotenv_path)

def main() -> None:
    """
    A command-line tool to inspect the contents of the Mnemonic Cortex ChromaDB.
    """
    # Resolve project root as a Path and load env
    project_root = Path(find_project_root())
    setup_environment(str(project_root))
    db_path = os.getenv("DB_PATH", "chroma_db")
    _env = os.getenv("CHROMA_ROOT", "").strip()
    # Prefer CHROMA_ROOT from .env (absolute or repo-relative); fall back to project layout
    CHROMA_ROOT = (Path(_env) if Path(_env).is_absolute() else (project_root / _env)).resolve() if _env else (project_root / 'mnemonic_cortex' / db_path)
    CHROMA_CHILD_COLLECTION = os.getenv("CHROMA_CHILD_COLLECTION", "")
    full_db_path = str(CHROMA_ROOT)

    if not os.path.exists(full_db_path):
        print(f"ERROR: Database not found at '{full_db_path}'. Please run the ingestion script first.")
        return

    # The ingestion script writes collection data into a child collection folder
    # (for example `child_chunks_v5`) under the `chroma_db` root. Historically
    # some scripts pointed to the collection folder directly. To be robust we
    # detect and use a child collection folder if one exists.
    print(f"--- Inspecting ChromaDB root at '{full_db_path}' ---")
    # If the path contains a child collection folder, prefer that one.
    chosen_path = full_db_path
    try:
        entries = [e for e in os.listdir(full_db_path) if os.path.isdir(os.path.join(full_db_path, e))]
    except Exception:
        entries = []

    # If user provided CHROMA_CHILD_COLLECTION in .env, prefer it.
    if CHROMA_CHILD_COLLECTION:
        candidate = CHROMA_CHILD_COLLECTION
        if candidate in entries:
            chosen_path = os.path.join(full_db_path, candidate)
            print(f"Using collection from .env: '{candidate}' — path '{chosen_path}'")
        else:
            # fall back to autodetect below
            print(f"CHROMA_CHILD_COLLECTION='{CHROMA_CHILD_COLLECTION}' set in .env but not found under '{full_db_path}'. Falling back to autodetect.")

    if entries and not CHROMA_CHILD_COLLECTION:
        # Prefer a directory that looks like a child_chunks collection, else pick first
        candidate = None
        for e in entries:
            if e.startswith("child_chunks"):
                candidate = e
                break
        if candidate is None:
            candidate = entries[0]
        chosen_path = os.path.join(full_db_path, candidate)
        print(f"Detected collection directory '{candidate}' — using '{chosen_path}' for inspection.")
    else:
        print(f"No child collection subdirectories detected under '{full_db_path}'; attempting to open the path directly.")

    embedding_model = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")

    # If the chosen_path is a collection directory like 'child_chunks_v5', the
    # langchain Chroma wrapper expects the collection name to match what was
    # used during ingestion. Use the folder name as the collection_name when
    # appropriate; otherwise open the DB at the root.
    collection_name = None
    base = os.path.basename(chosen_path)
    if base.startswith("child_chunks"):
        collection_name = base

    if collection_name:
        print(f"Opening Chroma collection '{collection_name}' at '{chosen_path}'")
        db = Chroma(persist_directory=chosen_path, embedding_function=embedding_model, collection_name=collection_name)
    else:
        print(f"Opening Chroma at '{chosen_path}' (no explicit collection_name)")
        db = Chroma(persist_directory=chosen_path, embedding_function=embedding_model)

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