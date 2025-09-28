"""
Ingestion Script (scripts/ingest.py) v2.3 - Archive Exclusion Hardened

This script implements the Ingestion Pipeline of the Mnemonic Cortex RAG system.
It processes the canonical Cognitive Genome by directly traversing the project directory
and ingesting each markdown file as a discrete, atomic source of truth, with explicit
exclusion of all 'archive' directories to prevent ingestion of legacy or malformed files.

Role in RAG Pipeline:
- Traverses specified project directories to find all .md files, excluding archives.
- Loads each file individually using TextLoader.
- Splits documents using markdown headers for semantic chunking.
- Embeds chunks with Nomic Embed model in local mode.
- Stores verifiable GitHub source URLs in metadata for each chunk.
- Persists the vector store to disk for use by the Query Pipeline.

Key Improvements in v2.3:
- Archive exclusion: Prevents ingestion of superseded or malformed files.
- Direct atomic ingestion: No longer relies on snapshot files.
- Verifiable sources: Each chunk includes a full GitHub URL in metadata.
- Superior architecture: Processes files individually for perfect traceability.

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

        print("Re-building vector store with canonical-only sources...")
        Chroma.from_documents(
            documents=all_splits,
            embedding=embedding_model,
            persist_directory=full_db_path
        )
        print(f"Successfully created and persisted final vector store at '{full_db_path}'.")
        print("--- Ingestion Process Complete ---")

    except Exception as e:
        print(f"\n--- INGESTION FAILED ---")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()