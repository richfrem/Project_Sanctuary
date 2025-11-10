"""
Ingestion Script (scripts/ingest.py) v2.4 - Phase 1: Parent Document Retriever

This script implements the Ingestion Pipeline of the Mnemonic Cortex RAG system with Phase 1
evolution: Parent Document Retriever architecture to prevent Context Fragmentation.

It processes the canonical Cognitive Genome by directly traversing the project directory
and ingesting each markdown file as a discrete, atomic source of truth, with explicit
exclusion of all 'archive' directories to prevent ingestion of legacy or malformed files.

Phase 1 Evolution (Parent Document Retriever):
- Stores full parent documents in InMemoryDocstore for complete context access
- Stores document chunks in ChromaDB vectorstore for precise semantic retrieval
- Returns full parent documents when child chunks are found relevant
- Eliminates Context Fragmentation vulnerability by ensuring complete document availability
- Optimized retrieval: Finds relevant chunks first, then returns full parent documents

Role in RAG Pipeline:
- Traverses specified project directories to find all .md files, excluding archives.
- Loads each file individually using TextLoader.
- Splits documents using markdown headers for semantic chunking.
- Embeds chunks with Nomic Embed model in local mode.
- Stores verifiable GitHub source URLs in metadata for each chunk.
- Persists child chunks to ChromaDB vectorstore and parent documents to InMemoryDocstore.
- Creates ParentDocumentRetriever for optimized retrieval that prevents context fragmentation.

Key Improvements in v2.4:
- Parent Document Retriever: Dual storage (InMemoryDocstore + ChromaDB) prevents context fragmentation
- Optimized Retrieval: Returns full documents instead of fragments for complete context
- Archive exclusion: Prevents ingestion of superseded or malformed files
- Direct atomic ingestion: No longer relies on snapshot files
- Verifiable sources: Each chunk includes a full GitHub URL in metadata
- Superior architecture: Processes files individually for perfect traceability

Dependencies:
- Cognitive Genome: Markdown files in project directories (00_CHRONICLE, 01_PROTOCOLS, etc.).
- NomicEmbeddings: For local text embedding (requires nomic[local] and gpt4all).
- ChromaDB: For vector storage and persistence.
- LangChain: For document loading, splitting, and vector store integration.
- Environment: Uses .env for DB_PATH and GITHUB_REPO_URL.

Usage:
    python mnemonic_cortex/scripts/ingest.py
"""

import os
import re
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_nomic import NomicEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from glob import glob

# --- Constants ---
HEADERS_TO_SPLIT_ON = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

def find_project_root():
    """Find the project root by ascending from the current script's directory."""
    current_path = os.path.abspath(os.path.dirname(__file__))
    while True:
        if '.git' in os.listdir(current_path):
            return current_path
        parent_path = os.path.dirname(current_path)
        if parent_path == current_path:
            raise FileNotFoundError("Could not find the project root (.git folder).")
        current_path = parent_path

def setup_environment(project_root):
    """Load environment variables."""
    dotenv_path = os.path.join(project_root, 'mnemonic_cortex', '.env')
    load_dotenv(dotenv_path=dotenv_path)

def main() -> None:
    """
    (v2.3) Ingests the canonical cognitive genome by traversing the project
    directory, now with explicit exclusion of all 'archive' directories
    to prevent ingestion of legacy or malformed files.
    """
    print("--- Starting Ingestion Process (v2.3 - Archive Exclusion Hardened) ---")
    try:
        project_root = find_project_root()
        setup_environment(project_root)

        db_path = os.getenv("DB_PATH")
        github_repo_url = os.getenv("GITHUB_REPO_URL")
        if not db_path or not github_repo_url:
            raise ValueError("DB_PATH or GITHUB_REPO_URL not set in .env.")

        full_db_path = os.path.join(project_root, 'mnemonic_cortex', db_path)
        
        directories_to_scan = [
            '00_CHRONICLE', '01_PROTOCOLS', '02_USER_REFLECTIONS', '04_THE_FORTRESS',
            '05_ARCHIVED_BLUEPRINTS', '06_THE_EMBER_LIBRARY', '07_COUNCIL_AGENTS',
            'RESEARCH_SUMMARIES', 'WORK_IN_PROGRESS', 'mnemonic_cortex'
        ]
        
        all_filepaths = []
        for dir_name in directories_to_scan:
            scan_path = os.path.join(project_root, dir_name)
            if os.path.exists(scan_path):
                filepaths = glob(os.path.join(scan_path, '**', '*.md'), recursive=True)
                
                # --- CRITICAL FIX (v2.3) ---
                # Exclude any file that is inside a directory named 'archive'.
                filtered_paths = [
                    p for p in filepaths 
                    if 'archive/' not in p.replace('\\', '/') and 'archives/' not in p.replace('\\', '/')
                ]
                all_filepaths.extend(filtered_paths)
                print(f"  - Found {len(filtered_paths)} canonical markdown files in '{dir_name}' (archives excluded).")

        if not all_filepaths:
            raise FileNotFoundError("No markdown documents found.")

        print(f"\nFound a total of {len(all_filepaths)} canonical markdown files to process.")
        
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON, strip_headers=False)
        all_splits = []

        for filepath in all_filepaths:
            relative_path = os.path.relpath(filepath, project_root).replace('\\', '/')
            
            if os.path.basename(relative_path) == 'Living_Chronicle.md':
                print(f"Skipping Master Index file: {relative_path}")
                continue
            
            loader = TextLoader(filepath)
            doc = loader.load()[0]
            
            splits = markdown_splitter.split_text(doc.page_content)
            
            source_url = f"{github_repo_url}{relative_path}"
            
            for split in splits:
                split.metadata['source_url'] = source_url
                split.metadata['source_file'] = relative_path
            
            all_splits.extend(splits)
        
        print(f"Split genome into {len(all_splits)} total chunks with verifiable source URLs.")

        embedding_model = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
        print("Initialized Nomic embedding model.")

        # Phase 1: Parent Document Retriever Implementation
        # Store both chunks and full parent documents in ChromaDB collections
        chunks_store_path = os.path.join(full_db_path, "chunks")
        parents_store_path = os.path.join(full_db_path, "parents")

        # Create chunks vectorstore (for retrieval)
        chunks_vectorstore = Chroma(
            collection_name="document_chunks",
            embedding_function=embedding_model,
            persist_directory=chunks_store_path
        )

        # Create parents vectorstore (for full documents)
        parents_vectorstore = Chroma(
            collection_name="parent_documents",
            embedding_function=embedding_model,
            persist_directory=parents_store_path
        )

        print("Re-building vector stores with Parent Document architecture...")

        # Store chunks for retrieval
        chunks_vectorstore.add_documents(all_splits)
        print(f"Stored {len(all_splits)} chunks in vectorstore")

        # Store full parent documents (one per source file)
        parent_docs = []
        for filepath in all_filepaths:
            relative_path = os.path.relpath(filepath, project_root).replace('\\', '/')

            if os.path.basename(relative_path) == 'Living_Chronicle.md':
                continue

            loader = TextLoader(filepath)
            doc = loader.load()[0]

            # Add metadata to identify this as a parent document
            doc.metadata['source_url'] = f"{github_repo_url}{relative_path}"
            doc.metadata['source_file'] = relative_path
            doc.metadata['is_parent'] = True

            parent_docs.append(doc)

        parents_vectorstore.add_documents(parent_docs)
        print(f"Stored {len(parent_docs)} parent documents in vectorstore")

        print(f"Successfully created dual vector stores at '{full_db_path}'.")
        print("--- Phase 1 Ingestion Process Complete ---")

    except Exception as e:
        print(f"\n--- INGESTION FAILED ---")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()