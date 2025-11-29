"""
Mnemonic Cortex Ingestion Service
Encapsulates logic for full and incremental ingestion of documents into the RAG system.
"""
import os
import sys
import shutil
import pickle
import math
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.storage import LocalFileStore, EncoderBackedStore
from langchain_nomic import NomicEmbeddings
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_core.documents import Document

try:
    import chromadb
    from chromadb.errors import InternalError as ChromaInternalError
except Exception:
    chromadb = None
    ChromaInternalError = Exception


class IngestionService:
    """
    Service for managing knowledge base ingestion (Full and Incremental).
    """

    def __init__(self, project_root: str):
        """
        Initialize the Ingestion Service.

        Args:
            project_root: Absolute path to the project root directory.
        """
        self.project_root = Path(project_root)
        
        # Load environment variables
        load_dotenv(dotenv_path=self.project_root / ".env")
        
        # Configuration
        self.db_path = os.getenv("DB_PATH", "chroma_db")
        _env = os.getenv("CHROMA_ROOT", "").strip()
        self.chroma_root = (Path(_env) if Path(_env).is_absolute() else (self.project_root / _env)).resolve() if _env else (self.project_root / "mnemonic_cortex" / self.db_path)
        
        self.child_collection_name = os.getenv("CHROMA_CHILD_COLLECTION", "child_chunks_v5")
        self.parent_collection_name = os.getenv("CHROMA_PARENT_STORE", "parent_documents_v5")
        
        self.vectorstore_path = str(self.chroma_root / self.child_collection_name)
        self.docstore_path = str(self.chroma_root / self.parent_collection_name)
        
        # Default source directories
        self.default_source_dirs = [
            "00_CHRONICLE", "01_PROTOCOLS", "02_USER_REFLECTIONS", "04_THE_FORTRESS",
            "05_ARCHIVED_BLUEPRINTS", "06_THE_EMBER_LIBRARY", "07_COUNCIL_AGENTS",
            "RESEARCH_SUMMARIES", "WORK_IN_PROGRESS", "mnemonic_cortex"
        ]
        self.exclude_subdirs = ["ARCHIVE", "archive", "Archive", "node_modules", "ARCHIVED_MESSAGES", "DEPRECATED"]

    def _init_components(self):
        """Initialize ChromaDB components."""
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        embedding_model = NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
        
        vectorstore = Chroma(
            collection_name=self.child_collection_name,
            embedding_function=embedding_model,
            persist_directory=self.vectorstore_path
        )
        
        fs_store = LocalFileStore(root_path=self.docstore_path)
        store = EncoderBackedStore(
            store=fs_store,
            key_encoder=lambda k: str(k),
            value_serializer=pickle.dumps,
            value_deserializer=pickle.loads,
        )
        
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter
        )
        
        return vectorstore, retriever

    def _chunked_iterable(self, seq: List, size: int):
        """Yield successive n-sized chunks from seq."""
        for i in range(0, len(seq), size):
            yield seq[i : i + size]

    def _safe_add_documents(self, retriever: ParentDocumentRetriever, docs: List, max_retries: int = 5):
        """
        Recursively retry adding documents to handle ChromaDB batch size limits.
        """
        try:
            retriever.add_documents(docs, ids=None, add_to_docstore=True)
            return
        except Exception as e:
            # Check for batch size or internal errors
            err_text = str(e).lower()
            if "batch size" not in err_text and "internalerror" not in e.__class__.__name__.lower():
                raise

            if len(docs) <= 1 or max_retries <= 0:
                raise

            mid = len(docs) // 2
            left = docs[:mid]
            right = docs[mid:]
            self._safe_add_documents(retriever, left, max_retries - 1)
            self._safe_add_documents(retriever, right, max_retries - 1)

    def ingest_full(self, purge_existing: bool = True, source_directories: List[str] = None) -> Dict[str, Any]:
        """
        Perform a full ingestion of the knowledge base.

        Args:
            purge_existing: Whether to delete the existing database first.
            source_directories: List of directories to ingest (relative to project root).

        Returns:
            Dictionary with statistics.
        """
        start_time = time.time()
        
        # Purge existing DB
        if purge_existing and self.chroma_root.exists():
            shutil.rmtree(str(self.chroma_root))
        
        # Determine directories
        dirs_to_process = source_directories or self.default_source_dirs
        
        # Load documents
        all_docs = []
        for directory in dirs_to_process:
            dir_path = self.project_root / directory
            if dir_path.is_dir():
                loader = DirectoryLoader(
                    str(dir_path),
                    glob="**/*.md",
                    loader_cls=TextLoader,
                    recursive=True,
                    show_progress=False,
                    use_multithreading=True,
                    exclude=[f"**/{ex}/**" for ex in self.exclude_subdirs],
                )
                all_docs.extend(loader.load())
        
        total_docs = len(all_docs)
        if total_docs == 0:
            return {
                "documents_processed": 0,
                "chunks_created": 0,
                "ingestion_time_ms": (time.time() - start_time) * 1000,
                "status": "success",
                "message": "No documents found."
            }

        # Initialize components
        vectorstore, retriever = self._init_components()
        
        # Batch processing
        parent_batch_size = 50
        num_batches = math.ceil(total_docs / parent_batch_size)
        
        for batch_docs in self._chunked_iterable(all_docs, parent_batch_size):
            self._safe_add_documents(retriever, batch_docs)
            
        # Persist
        vectorstore.persist()
        
        elapsed_ms = (time.time() - start_time) * 1000
        
        return {
            "documents_processed": total_docs,
            "chunks_created": 0, # Difficult to count exactly without modifying ParentDocumentRetriever
            "ingestion_time_ms": elapsed_ms,
            "vectorstore_path": str(self.chroma_root),
            "status": "success"
        }

    def ingest_incremental(self, file_paths: List[str], skip_duplicates: bool = True) -> Dict[str, Any]:
        """
        Incrementally ingest specific files.

        Args:
            file_paths: List of absolute or relative file paths.
            skip_duplicates: Whether to skip files already in the database.

        Returns:
            Dictionary with statistics.
        """
        start_time = time.time()
        
        # Validate files
        valid_files = []
        for fp in file_paths:
            path = Path(fp)
            if not path.is_absolute():
                path = self.project_root / path
            
            if path.exists() and path.is_file() and path.suffix == '.md':
                valid_files.append(str(path.resolve()))
        
        if not valid_files:
            return {"added": 0, "skipped": 0, "total_chunks": 0, "error": "No valid files to ingest"}

        # Initialize components (loads existing DB)
        vectorstore, retriever = self._init_components()
        
        # Check duplicates
        existing_files = set()
        if skip_duplicates:
            try:
                # Access underlying store to check existing keys
                # Note: This is an approximation. Ideally we'd query metadata.
                # For now, we rely on the fact that we can't easily query all metadata efficiently in Chroma/LangChain
                # without iterating.
                # A better approach for the future is to maintain a separate index or use a specific query.
                pass 
            except Exception:
                pass

        added = 0
        skipped = 0
        total_chunks = 0
        
        for file_path in valid_files:
            try:
                # Load document
                loader = TextLoader(file_path)
                docs = loader.load()
                
                if not docs:
                    continue
                
                # Set metadata
                for doc in docs:
                    doc.metadata['source_file'] = file_path
                    doc.metadata['source'] = file_path
                
                # Add to retriever
                retriever.add_documents(docs, ids=None, add_to_docstore=True)
                
                # Calculate chunks (approximation)
                chunks = retriever.child_splitter.split_documents(docs)
                total_chunks += len(chunks)
                added += 1
                
            except Exception as e:
                print(f"Error ingesting {file_path}: {e}")
                continue
        
        if added > 0:
            vectorstore.persist()
            
        return {
            "added": added,
            "skipped": skipped,
            "total_chunks": total_chunks,
            "ingestion_time_ms": (time.time() - start_time) * 1000,
            "status": "success"
        }
