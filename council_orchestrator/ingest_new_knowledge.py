#!/usr/bin/env python3
import sys
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions

# --- PATCH START ---
# Define the project root as an absolute anchor, based on this script's location.
# This makes the script independent of the caller's current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent
# --- PATCH END ---

# Configure your ChromaDB path and collection name
CHROMA_PATH = PROJECT_ROOT / "mnemonic_cortex/chroma_db" # Correctly anchored path
COLLECTION_NAME = "sanctuary_cortex"

# Use the same embedding function as your main script
EMBEDDING_FUNC = embedding_functions.DefaultEmbeddingFunction()

def main(file_path_str: str):
    file_path = Path(file_path_str)
    if not file_path.is_absolute():
        # --- PATCH START ---
        # Correctly resolve relative paths from the project root, not the cwd.
        file_path = PROJECT_ROOT / file_path
        # --- PATCH END ---

    if not file_path.exists():
        print(f"[INGEST ERROR] File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[INGEST] Processing file: {file_path.name}")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=EMBEDDING_FUNC
    )

    content = file_path.read_text(encoding="utf-8")
    # --- PATCH START ---
    # Ensure the document ID is consistently relative to the project root.
    doc_id = str(file_path.relative_to(PROJECT_ROOT))
    # --- PATCH END ---

    # Add the new document to the collection
    collection.add(
        documents=[content],
        metadatas=[{"source": file_path.name}],
        ids=[doc_id]
    )
    print(f"[INGEST SUCCESS] Document '{doc_id}' added to Mnemonic Cortex.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python ingest_new_knowledge.py <path/to/file.md>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])