"""mnemonic_cortex/scripts/ingest_incremental.py

Incremental ingestion script for adding individual documents to the Mnemonic Cortex
without rebuilding the entire database.

Usage:
    python3 ingest_incremental.py file1.md file2.md ...
    python3 ingest_incremental.py --help

Features:
- Loads existing ChromaDB collections (no purge)
- Adds new documents incrementally
- Skips duplicates based on source_file metadata
- Returns statistics (added, skipped, total chunks)
"""
import os
import sys
import pickle
import argparse
from pathlib import Path
from typing import List, Tuple
from dotenv import load_dotenv

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Load repo-root .env
load_dotenv(dotenv_path=project_root / ".env")

# Imports
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.storage import LocalFileStore, EncoderBackedStore
from langchain_nomic import NomicEmbeddings
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document

# Configuration
DB_PATH = os.getenv("DB_PATH", "chroma_db")
_env = os.getenv("CHROMA_ROOT", "").strip()
CHROMA_ROOT = (Path(_env) if Path(_env).is_absolute() else (project_root / _env)).resolve() if _env else (project_root / "mnemonic_cortex" / DB_PATH)
CHILD_COLLECTION = os.getenv("CHROMA_CHILD_COLLECTION", "child_chunks_v5")
PARENT_COLLECTION = os.getenv("CHROMA_PARENT_STORE", "parent_documents_v5")
VECTORSTORE_PATH = str(CHROMA_ROOT / CHILD_COLLECTION)
DOCSTORE_PATH = str(CHROMA_ROOT / PARENT_COLLECTION)


def load_existing_collections() -> Tuple[Chroma, EncoderBackedStore, ParentDocumentRetriever]:
    """Load existing ChromaDB collections without purging."""
    print(f"Loading existing collections from {CHROMA_ROOT}")
    
    # Initialize embedding model
    embedding_model = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
    
    # Initialize text splitter
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    
    # Load existing vectorstore
    vectorstore = Chroma(
        collection_name=CHILD_COLLECTION,
        embedding_function=embedding_model,
        persist_directory=VECTORSTORE_PATH
    )
    
    # Load existing docstore
    fs_store = LocalFileStore(root_path=DOCSTORE_PATH)
    docstore = EncoderBackedStore(
        store=fs_store,
        key_encoder=lambda k: str(k),
        value_serializer=pickle.dumps,
        value_deserializer=pickle.loads,
    )
    
    # Initialize retriever
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        child_splitter=child_splitter
    )
    
    return vectorstore, docstore, retriever


def get_existing_source_files(docstore: EncoderBackedStore) -> set:
    """Get set of source files already in the docstore."""
    existing_files = set()
    
    # Iterate through docstore to find existing source files
    # Note: This is a simple implementation. For large databases, consider
    # maintaining a separate index of source files.
    try:
        # Access the underlying LocalFileStore
        fs_store = docstore.store
        for key in fs_store.yield_keys():
            # Keys are document IDs, we need to load and check metadata
            try:
                doc = docstore.mget([key])[0]
                if doc and hasattr(doc, 'metadata'):
                    source_file = doc.metadata.get('source_file')
                    if source_file:
                        existing_files.add(source_file)
            except Exception:
                continue
    except Exception as e:
        print(f"Warning: Could not enumerate existing files: {e}")
    
    return existing_files


def ingest_files(file_paths: List[str], skip_duplicates: bool = True) -> dict:
    """
    Incrementally ingest files into the Mnemonic Cortex.
    
    Args:
        file_paths: List of file paths to ingest
        skip_duplicates: Whether to skip files already in the database
        
    Returns:
        Dictionary with statistics (added, skipped, total_chunks)
    """
    # Validate files exist
    valid_files = []
    for fp in file_paths:
        path = Path(fp)
        if not path.exists():
            print(f"Warning: File not found: {fp}")
            continue
        if not path.is_file():
            print(f"Warning: Not a file: {fp}")
            continue
        if not fp.endswith('.md'):
            print(f"Warning: Not a markdown file: {fp}")
            continue
        valid_files.append(str(path.resolve()))
    
    if not valid_files:
        return {"added": 0, "skipped": 0, "total_chunks": 0, "error": "No valid files to ingest"}
    
    # Load existing collections
    vectorstore, docstore, retriever = load_existing_collections()
    
    # Get existing source files if skipping duplicates
    existing_files = set()
    if skip_duplicates:
        print("Checking for existing documents...")
        existing_files = get_existing_source_files(docstore)
        print(f"Found {len(existing_files)} existing documents")
    
    # Process files
    added = 0
    skipped = 0
    total_chunks = 0
    
    for file_path in valid_files:
        # Check if already exists
        if skip_duplicates and file_path in existing_files:
            print(f"Skipping duplicate: {file_path}")
            skipped += 1
            continue
        
        # Load document
        try:
            loader = TextLoader(file_path)
            docs = loader.load()
            
            if not docs:
                print(f"Warning: No content loaded from {file_path}")
                continue
            
            # Set metadata
            for doc in docs:
                doc.metadata['source_file'] = file_path
                doc.metadata['source'] = file_path
            
            # Add to retriever
            print(f"Ingesting: {file_path}")
            retriever.add_documents(docs, ids=None, add_to_docstore=True)
            
            # Calculate chunks
            chunks = retriever.child_splitter.split_documents(docs)
            chunk_count = len(chunks)
            
            added += 1
            total_chunks += chunk_count
            print(f"  ✓ Added {chunk_count} chunks")
            
        except Exception as e:
            print(f"Error ingesting {file_path}: {e}")
            continue
    
    # Persist vectorstore
    if added > 0:
        print("Persisting vectorstore...")
        vectorstore.persist()
    
    return {
        "added": added,
        "skipped": skipped,
        "total_chunks": total_chunks
    }


def main():
    parser = argparse.ArgumentParser(
        description="Incrementally ingest documents into the Mnemonic Cortex"
    )
    parser.add_argument(
        "files",
        nargs="+",
        help="Markdown files to ingest"
    )
    parser.add_argument(
        "--no-skip-duplicates",
        action="store_true",
        help="Do not skip duplicate files (re-ingest)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Mnemonic Cortex - Incremental Ingestion")
    print("=" * 60)
    print(f"Files to process: {len(args.files)}")
    print()
    
    # Run ingestion
    stats = ingest_files(args.files, skip_duplicates=not args.no_skip_duplicates)
    
    # Print results
    print()
    print("=" * 60)
    print("Ingestion Complete")
    print("=" * 60)
    print(f"Documents added: {stats['added']}")
    print(f"Documents skipped: {stats['skipped']}")
    print(f"Total chunks created: {stats['total_chunks']}")
    
    if 'error' in stats:
        print(f"Error: {stats['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()
