"""mnemonic_cortex/scripts/ingest.py

Batch-aware ingestion script that splits parent documents into manageable
batches and avoids ChromaDB's max-batch limits.
"""
import os
import sys
import shutil
import pickle
import math
from pathlib import Path
from dotenv import load_dotenv
from typing import List

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Load repo-root .env so CHROMA_ROOT/DB_PATH and collection names are available
load_dotenv(dotenv_path=project_root / ".env")

# Working imports (adapted to installed langchain packages in this environment)
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.storage import LocalFileStore, EncoderBackedStore
from langchain_nomic import NomicEmbeddings
from langchain_classic.retrievers import ParentDocumentRetriever

try:
    import chromadb
    from chromadb.errors import InternalError as ChromaInternalError
except Exception:
    chromadb = None
    ChromaInternalError = Exception

# --- CONFIGURATION (v5 for final architecture) ---
# Respect CHROMA_ROOT from repo .env when present, otherwise fall back to
# the historical project layout (mnemonic_cortex/DB_PATH).
DB_PATH = os.getenv("DB_PATH", "chroma_db")
_env = os.getenv("CHROMA_ROOT", "").strip()
CHROMA_ROOT = (Path(_env) if Path(_env).is_absolute() else (project_root / _env)).resolve() if _env else (project_root / "mnemonic_cortex" / DB_PATH)
# Collection names are now configurable via env vars so we don't hardcode v4/v5.
CHILD_COLLECTION = os.getenv("CHROMA_CHILD_COLLECTION", "child_chunks_v5")
PARENT_COLLECTION = os.getenv("CHROMA_PARENT_STORE", "parent_documents_v5")
VECTORSTORE_PATH = str(CHROMA_ROOT / CHILD_COLLECTION)
DOCSTORE_PATH = str(CHROMA_ROOT / PARENT_COLLECTION)
SOURCE_DIRECTORIES = [
    "00_CHRONICLE", "01_PROTOCOLS", "02_USER_REFLECTIONS", "04_THE_FORTRESS",
    "05_ARCHIVED_BLUEPRINTS", "06_THE_EMBER_LIBRARY", "07_COUNCIL_AGENTS",
    "RESEARCH_SUMMARIES", "WORK_IN_PROGRESS", "mnemonic_cortex"
]
EXCLUDE_SUBDIRS = ["ARCHIVE", "archive", "Archive", "node_modules", "ARCHIVED_MESSAGES", "DEPRECATED"]


def chunked_iterable(seq: List, size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def safe_add_documents(retriever: ParentDocumentRetriever, docs: List, max_retries: int = 5):
    """Call retriever.add_documents but retry by subdividing the batch on chroma overflow.

    This function will try to add `docs` as a single batch. If Chroma raises an
    internal batch-size error, it will split the batch into two and retry
    recursively until success or until max_retries is reached.
    """
    try:
        retriever.add_documents(docs, ids=None, add_to_docstore=True)
        return
    except Exception as e:  # catch chromadb.errors.InternalError and others
        # If it's not obviously a batch-size/internal error, re-raise after a few tries
        err_text = str(e).lower()
        if "batch size" not in err_text and "internalerror" not in e.__class__.__name__.lower():
            raise

        if len(docs) <= 1 or max_retries <= 0:
            # give up and re-raise for single-document failure or retries exhausted
            raise

        mid = len(docs) // 2
        left = docs[:mid]
        right = docs[mid:]
        safe_add_documents(retriever, left, max_retries - 1)
        safe_add_documents(retriever, right, max_retries - 1)


def main():
    print("--- Starting Ingestion Process (Disciplined Batch Architecture) ---")
    # Purge any existing DB root so we start clean
    if CHROMA_ROOT.exists():
        print(f"Purging existing database at {CHROMA_ROOT}")
        shutil.rmtree(str(CHROMA_ROOT))

    # 1. Load documents
    all_docs = []
    for directory in SOURCE_DIRECTORIES:
        dir_path = project_root / directory
        if dir_path.is_dir():
            loader = DirectoryLoader(
                str(dir_path),
                glob="**/*.md",
                loader_cls=TextLoader,
                recursive=True,
                show_progress=False,
                use_multithreading=True,
                exclude=[f"**/{ex}/**" for ex in EXCLUDE_SUBDIRS],
            )
            all_docs.extend(loader.load())
    total_docs = len(all_docs)
    print(f"Found a total of {total_docs} canonical markdown files to process.")

    if total_docs == 0:
        print("No documents found. Exiting.")
        return

    # 2. Initialize components
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    embedding_model = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
    vectorstore = Chroma(collection_name=CHILD_COLLECTION, embedding_function=embedding_model, persist_directory=VECTORSTORE_PATH)
    fs_store = LocalFileStore(root_path=DOCSTORE_PATH)
    store = EncoderBackedStore(
        store=fs_store,
        key_encoder=lambda k: str(k),
        value_serializer=pickle.dumps,
        value_deserializer=pickle.loads,
    )
    retriever = ParentDocumentRetriever(vectorstore=vectorstore, docstore=store, child_splitter=child_splitter)

    # 3. Batch processing parameters
    parent_batch_size = 50  # number of parent documents per batch (safe default)
    num_batches = math.ceil(total_docs / parent_batch_size)

    print(f"Adding {total_docs} documents in {num_batches} batches of up to {parent_batch_size} parents each...")

    for batch_idx, batch_docs in enumerate(chunked_iterable(all_docs, parent_batch_size), start=1):
        print(f"  - Processing batch {batch_idx}/{num_batches} with {len(batch_docs)} parent docs...")
        try:
            safe_add_documents(retriever, batch_docs)
        except Exception as e:
            print(f"Failed to add batch {batch_idx}: {e}")
            raise

    print("All batches processed. Persisting vector store...")
    vectorstore.persist()
    print("--- Ingestion Process Complete ---")


if __name__ == "__main__":
    main()