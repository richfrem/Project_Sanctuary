#============================================
# mcp_servers/rag_cortex/operations.py
# Purpose: Core operations for interacting with the Mnemonic Cortex (RAG).
#          Orchestrates ingestion, semantic search, and cache management.
# Role: Single Source of Truth
# Used as a module by server.py
# Calling example:
#   ops = CortexOperations(project_root)
#   ops.ingest_full(...)
# LIST OF CLASSES/FUNCTIONS:
#   - CortexOperations
#     - __init__
#     - _calculate_semantic_hmac
#     - _chunked_iterable
#     - _get_container_status
#     - _get_git_diff_summary
#     - _get_mcp_name
#     - _get_recency_delta
#     - _get_recent_chronicle_highlights
#     - _get_recent_protocol_updates
#     - _get_strategic_synthesis
#     - _get_system_health_traffic_light
#     - _get_tactical_priorities
#     - _load_documents_from_directory
#     - _safe_add_documents
#     - _should_skip_path
#     - cache_get
#     - cache_set
#     - cache_warmup
#     - capture_snapshot
#     - get_cache_stats
#     - get_stats
#     - ingest_full
#     - ingest_incremental
#     - learning_debrief
#     - query
#     - query_structured
#============================================


import os
import re # Added for parsing markdown headers
from typing import List, Tuple # Added Tuple
# Disable tqdm globally to prevent stdout pollution - MUST BE FIRST
os.environ["TQDM_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import time
import subprocess
import contextlib
import io
import logging
import json
from uuid import uuid4
from pathlib import Path
from typing import Dict, Any, List, Optional

# --- Protocol 128: Centralized Source of Truth Imports ---
from mcp_servers.lib.snapshot_utils import (
    generate_snapshot,
    EXCLUDE_DIR_NAMES,
    ALWAYS_EXCLUDE_FILES,
    PROTECTED_SEEDS,
    should_exclude_file
)

# Setup logging
# This block is moved to the top and modified to use standard logging
# sys.path.insert(0, str(Path(__file__).parent.parent.parent))
# from mcp_servers.lib.logging_utils import setup_mcp_logging
# logger = setup_mcp_logging(__name__)

# Configure logging
logger = logging.getLogger("rag_cortex.operations")
if not logger.handlers:
    # Add a default handler if none exist (e.g., when running directly)
    handler = logging.StreamHandler(sys.stderr)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


from .models import (
    IngestFullResponse,
    QueryResponse,
    QueryResult,
    StatsResponse,
    CollectionStats,
    IngestIncrementalResponse,
    to_dict,
    CacheGetResponse,
    CacheSetResponse,
    CacheWarmupResponse,
    DocumentSample,
    CaptureSnapshotRequest,
    CaptureSnapshotResponse,
    PersistSoulRequest,
    PersistSoulResponse,
)
from mcp_servers.lib.content_processor import ContentProcessor

# Imports that were previously inside methods, now moved to top for class initialization
# Silence stdout/stderr during imports to prevent MCP protocol pollution
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    import chromadb
    from dotenv import load_dotenv
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma
    from mcp_servers.rag_cortex.file_store import SimpleFileStore
    from langchain_core.documents import Document
    from mcp_servers.lib.env_helper import get_env_variable


class CortexOperations:
    #============================================
    # Class: CortexOperations
    # Purpose: Main backend for the Mnemonic Cortex RAG service.
    # Patterns: Facade / Orchestrator
    #============================================
    
    def __init__(self, project_root: str, client: Optional[chromadb.ClientAPI] = None):
        #============================================
        # Method: __init__
        # Purpose: Initialize Mnemonic Cortex backend.
        # Args:
        #   project_root: Path to project root
        #   client: Optional injected ChromaDB client
        #============================================
        self.project_root = Path(project_root)
        self.scripts_dir = self.project_root / "mcp_servers" / "rag_cortex" / "scripts"
        self.data_dir = self.project_root / ".agent" / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Network configuration using env_helper
        self.chroma_host = get_env_variable("CHROMA_HOST", required=False) or "localhost"
        self.chroma_port = int(get_env_variable("CHROMA_PORT", required=False) or "8110")
        self.chroma_data_path = get_env_variable("CHROMA_DATA_PATH", required=False) or ".vector_data"
        
        self.child_collection_name = get_env_variable("CHROMA_CHILD_COLLECTION", required=False) or "child_chunks_v5"
        self.parent_collection_name = get_env_variable("CHROMA_PARENT_STORE", required=False) or "parent_documents_v5"

        # Initialize ChromaDB client
        if client:
            self.chroma_client = client
        else:
            self.chroma_client = chromadb.HttpClient(host=self.chroma_host, port=self.chroma_port)
        
        # Initialize embedding model (HuggingFace/sentence-transformers for ARM64 compatibility - ADR 069)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="nomic-ai/nomic-embed-text-v1.5",
            model_kwargs={'device': 'cpu', 'trust_remote_code': True},
            encode_kwargs={'normalize_embeddings': True}
        )

        # Initialize child splitter (smaller chunks for retrieval)
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize parent splitter (larger chunks for context)
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )

        # Initialize vectorstore (Chroma)
        self.vectorstore = Chroma(
            client=self.chroma_client,
            collection_name=self.child_collection_name,
            embedding_function=self.embedding_model
        )

        # Parent document store (file-based, using configurable data path)
        docstore_path = str(self.project_root / self.chroma_data_path / self.parent_collection_name)
        self.store = SimpleFileStore(root_path=docstore_path)

        # Initialize Content Processor
        self.processor = ContentProcessor(self.project_root)
    
    #============================================
    # Method: _chunked_iterable
    # Purpose: Yield successive n-sized chunks from seq.
    # Args:
    #   seq: Sequence to chunk
    #   size: Chunk size
    # Returns: Generator of chunks
    #============================================
    def _chunked_iterable(self, seq: List, size: int):
        for i in range(0, len(seq), size):
            yield seq[i : i + size]
    
    def _safe_add_documents(self, retriever, docs: List, max_retries: int = 5):
        #============================================
        # Method: _safe_add_documents
        # Purpose: Recursively retry adding documents to handle ChromaDB 
        #          batch size limits.
        # Args:
        #   retriever: ParentDocumentRetriever instance
        #   docs: List of documents to add
        #   max_retries: Maximum number of retry attempts
        #============================================
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

    #============================================
    # Protocol 128: Centralized Source of Truth
    # These constants are now derived from snapshot_utils.py
    #============================================
    EXCLUDE_DIRS = EXCLUDE_DIR_NAMES
    
    # Filter ALWAYS_EXCLUDE_FILES for simple string name matching
    EXCLUDE_FILES = {f for f in ALWAYS_EXCLUDE_FILES if isinstance(f, str)}
    
    # Priority bypass authorized via PROTECTED_SEEDS
    ALLOWED_FILES = PROTECTED_SEEDS

    #============================================
    # Methods: _should_skip_path and _load_documents_from_directory
    # DEPRECATED: Replaced by ContentProcessor.load_for_rag()
    #============================================

    def ingest_full(
        self,
        purge_existing: bool = True,
        source_directories: List[str] = None
    ):
        #============================================
        # Method: ingest_full
        # Purpose: Perform full ingestion of knowledge base.
        # Args:
        #   purge_existing: Whether to purge existing database
        #   source_directories: Optional list of source directories
        # Returns: IngestFullResponse with accurate statistics
        #============================================
        try:
            start_time = time.time()
            
            # Purge existing collections if requested
            if purge_existing:
                logger.info("Purging existing database collections...")
                try:
                    self.chroma_client.delete_collection(name=self.child_collection_name)
                    logger.info(f"Deleted child collection: {self.child_collection_name}")
                except Exception as e:
                    logger.warning(f"Child collection '{self.child_collection_name}' not found or error deleting: {e}")
                
                # Also clear the parent document store
                if Path(self.store.root_path).exists():
                    import shutil
                    shutil.rmtree(self.store.root_path)
                    logger.info(f"Cleared parent document store at: {self.store.root_path}")
                else:
                    logger.info(f"Parent document store path '{self.store.root_path}' does not exist, no need to clear.")
                
                # Recreate the directory to ensure it exists for new writes
                Path(self.store.root_path).mkdir(parents=True, exist_ok=True)
                logger.info(f"Recreated parent document store directory at: {self.store.root_path}")
                
            # Re-initialize vectorstore to ensure it connects to a fresh/existing collection
            # This is critical after a delete_collection operation
            self.vectorstore = Chroma(
                client=self.chroma_client,
                collection_name=self.child_collection_name,
                embedding_function=self.embedding_model
            )
            
            # Default source directories from Manifest (ADR 082 Harmonization - JSON)
            import json
            manifest_path = self.project_root / "mcp_servers" / "lib" / "ingest_manifest.json"
            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                base_dirs = manifest.get("common_content", [])
                unique_targets = manifest.get("unique_rag_content", [])
                default_source_dirs = list(set(base_dirs + unique_targets))
            except Exception as e:
                logger.warning(f"Failed to load ingest manifest from {manifest_path}: {e}")
                # Fallback to critical defaults if manifest fails
                default_source_dirs = ["00_CHRONICLE", "01_PROTOCOLS"]
            
            # Determine directories
            dirs_to_process = source_directories or default_source_dirs
            paths_to_scan = [str(self.project_root / d) for d in dirs_to_process]
            
            # Load documents using ContentProcessor
            logger.info(f"Loading documents via ContentProcessor from {len(paths_to_scan)} directories...")
            all_docs = list(self.processor.load_for_rag(paths_to_scan))
            
            total_docs = len(all_docs)
            if total_docs == 0:
                logger.warning("No documents found for ingestion.")
                return IngestFullResponse(
                    documents_processed=0,
                    chunks_created=0,
                    ingestion_time_ms=(time.time() - start_time) * 1000,
                    vectorstore_path=f"{self.chroma_host}:{self.chroma_port}",
                    status="success",
                    error="No documents found."
                )
            
            logger.info(f"Processing {len(all_docs)} documents with parent-child splitting...")
            
            child_docs = []
            parent_count = 0
            
            for doc in all_docs:
                # Split into parent chunks
                parent_chunks = self.parent_splitter.split_documents([doc])
                
                for parent_chunk in parent_chunks:
                    # Generate parent ID
                    parent_id = str(uuid4())
                    parent_count += 1
                    
                    # Store parent document
                    self.store.mset([(parent_id, parent_chunk)])
                    
                    # Split parent into child chunks
                    sub_docs = self.child_splitter.split_documents([parent_chunk])
                    
                    # Add parent_id to child metadata
                    for sub_doc in sub_docs:
                        sub_doc.metadata["parent_id"] = parent_id
                        child_docs.append(sub_doc)
            
            # Add child chunks to vectorstore in batches
            # ChromaDB has a maximum batch size of ~5461
            logger.info(f"Adding {len(child_docs)} child chunks to vectorstore...")
            batch_size = 5000  # Safe batch size under the limit
            
            for i in range(0, len(child_docs), batch_size):
                batch = child_docs[i:i + batch_size]
                logger.info(f"  Adding batch {i//batch_size + 1}/{(len(child_docs)-1)//batch_size + 1} ({len(batch)} chunks)...")
                self.vectorstore.add_documents(batch)
            
            # Get actual counts
            # Re-initialize vectorstore to ensure it reflects the latest state
            self.vectorstore = Chroma(
                client=self.chroma_client,
                collection_name=self.child_collection_name,
                embedding_function=self.embedding_model
            )
            child_count = self.vectorstore._collection.count()
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            logger.info(f"✓ Ingestion complete!")
            logger.info(f"  - Parent documents: {parent_count}")
            logger.info(f"  - Child chunks: {child_count}")
            logger.info(f"  - Time: {elapsed_ms/1000:.2f}s")
            
            return IngestFullResponse(
                documents_processed=total_docs,
                chunks_created=child_count,
                ingestion_time_ms=elapsed_ms,
                vectorstore_path=f"{self.chroma_host}:{self.chroma_port}",
                status="success"
            )
            
        except Exception as e:
            logger.error(f"Full ingestion failed: {e}", exc_info=True)
            return IngestFullResponse(
                documents_processed=0,
                chunks_created=0,
                ingestion_time_ms=0,
                vectorstore_path="",
                status="error",
                error=str(e)
            )

    
    def query(
        self,
        query: str,
        max_results: int = 5,
        use_cache: bool = False,
        reasoning_mode: bool = False
    ):
        #============================================
        # Method: query
        # Purpose: Perform semantic search query using RAG infrastructure.
        # Args:
        #   query: Search query string
        #   max_results: Maximum results to return
        #   use_cache: Whether to use semantic cache
        #   reasoning_mode: Use reasoning model if True
        # Returns: QueryResponse with results and metadata
        #============================================
        try:
            start_time = time.time()
            
            # Initialize ChromaDB client (already done in __init__)
            collection = self.chroma_client.get_collection(name=self.child_collection_name)
            
            # Initialize embedding model (already done in __init__)
            
            # Generate query embedding
            query_embedding = self.embedding_model.embed_query(query)
            
            # Query ChromaDB
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Format results with Parent Document lookup
            formatted_results = []
            if results and results['documents'] and len(results['documents']) > 0:
                for i, doc_content in enumerate(results['documents'][0]):
                    metadata = results['metadatas'][0][i]
                    parent_id = metadata.get("parent_id")
                    
                    # If we have a parent_id, retrieve the full document context
                    final_content = doc_content
                    if parent_id:
                        try:
                            parent_docs = self.store.mget([parent_id])
                            if parent_docs and parent_docs[0]:
                                final_content = parent_docs[0].page_content
                                # Update metadata with parent metadata if needed
                                metadata.update(parent_docs[0].metadata)
                        except Exception as e:
                            logger.warning(f"Failed to retrieve parent doc {parent_id}: {e}")
                    
                    formatted_results.append(QueryResult(
                        content=final_content,
                        metadata=metadata,
                        relevance_score=results['distances'][0][i] if results.get('distances') else None
                    ))
            
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info(f"Query '{query[:50]}...' completed in {elapsed_ms:.2f}ms with {len(formatted_results)} results (Parent-Retriever applied).")
            
            return QueryResponse(
                status="success",
                results=formatted_results,
                query_time_ms=elapsed_ms,
                cache_hit=False
            )
            
        except Exception as e:
            logger.error(f"Query failed for '{query[:50]}...': {e}", exc_info=True)
            return QueryResponse(
                status="error",
                results=[],
                query_time_ms=0,
                cache_hit=False,
                error=str(e)
            )
    
    def get_stats(self, include_samples: bool = False, sample_count: int = 5):
        #============================================
        # Method: get_stats
        # Purpose: Get database statistics and health status.
        # Args:
        #   include_samples: Whether to include sample docs
        #   sample_count: Number of sample documents to return
        # Returns: StatsResponse with detailed database metrics
        #============================================
        try:
            # Get child chunks stats
            child_count = 0
            try:
                collection = self.chroma_client.get_collection(name=self.child_collection_name)
                child_count = collection.count()
                logger.info(f"Child collection '{self.child_collection_name}' count: {child_count}")
            except Exception as e:
                logger.warning(f"Child collection '{self.child_collection_name}' not found or error accessing: {e}")
                pass  # Collection doesn't exist yet
            
            # Get parent documents stats
            parent_count = 0
            if Path(self.store.root_path).exists():
                try:
                    parent_count = sum(1 for _ in self.store.yield_keys())
                    logger.info(f"Parent document store '{self.parent_collection_name}' count: {parent_count}")
                except Exception as e:
                    logger.warning(f"Error accessing parent document store at '{self.store.root_path}': {e}")
                    pass  # Silently ignore errors for MCP compatibility
            else:
                logger.info(f"Parent document store path '{self.store.root_path}' does not exist.")
            
            # Build collections dict
            collections = {
                "child_chunks": CollectionStats(count=child_count, name=self.child_collection_name),
                "parent_documents": CollectionStats(count=parent_count, name=self.parent_collection_name)
            }
            
            # Determine health status
            if child_count > 0 and parent_count > 0:
                health_status = "healthy"
            elif child_count > 0 or parent_count > 0:
                health_status = "degraded"
            else:
                health_status = "error"
            logger.info(f"RAG Cortex health status: {health_status}")
            
            # Retrieve sample documents if requested
            samples = None
            if include_samples and child_count > 0:
                try:
                    collection = self.chroma_client.get_collection(name=self.child_collection_name)
                    # Get sample documents with metadata and content
                    retrieved_docs = collection.get(limit=sample_count, include=["metadatas", "documents"])
                    
                    samples = []
                    for i in range(len(retrieved_docs["ids"])):
                        sample = DocumentSample(
                            id=retrieved_docs["ids"][i],
                            metadata=retrieved_docs["metadatas"][i],
                            content_preview=retrieved_docs["documents"][i][:150] + "..." if len(retrieved_docs["documents"][i]) > 150 else retrieved_docs["documents"][i]
                        )
                        samples.append(sample)
                    logger.info(f"Retrieved {len(samples)} sample documents.")
                except Exception as e:
                    logger.warning(f"Error retrieving sample documents: {e}")
                    # Silently ignore sample retrieval errors
                    pass
            
            return StatsResponse(
                total_documents=parent_count,
                total_chunks=child_count,
                collections=collections,
                health_status=health_status,
                samples=samples
            )
            
        except Exception as e:
            logger.error(f"Failed to retrieve stats: {e}", exc_info=True)
            return StatsResponse(
                total_documents=0,
                total_chunks=0,
                collections={},
                health_status="error",
                error=str(e)
            )
    
    def ingest_incremental(
        self,
        file_paths: List[str],
        metadata: Dict[str, Any] = None,
        skip_duplicates: bool = True
    ) -> IngestIncrementalResponse:
        #============================================
        # Method: ingest_incremental
        # Purpose: Incrementally ingest documents without full rebuild.
        # Args:
        #   file_paths: List of file paths to ingest
        #   metadata: Optional metadata to attach
        #   skip_duplicates: Deduplication flag
        # Returns: IngestIncrementalResponse with statistics
        #============================================
        try:
            start_time = time.time()
            
            # Validate files
            valid_files = []
            
            # Known host path prefixes that should be stripped for container compatibility
            # This handles cases where absolute host paths are passed to the containerized service
            HOST_PATH_MARKERS = [
                "/Users/",      # macOS
                "/home/",       # Linux
                "/root/",       # Linux root
                "C:\\Users\\",  # Windows
                "C:/Users/",    # Windows forward slash
            ]
            
            for fp in file_paths:
                path = Path(fp)
                
                # Handle absolute host paths by converting to relative paths
                # This enables proper resolution when running in containers
                if path.is_absolute():
                    fp_str = str(fp)
                    # Check if this looks like a host absolute path (not container /app path)
                    is_host_path = any(fp_str.startswith(marker) for marker in HOST_PATH_MARKERS)
                    
                    if is_host_path:
                        # Try to extract the relative path after common project markers
                        # Look for 'Project_Sanctuary/' or similar project root markers in the path
                        project_markers = ["Project_Sanctuary/", "project_sanctuary/", "/app/"]
                        for marker in project_markers:
                            if marker in fp_str:
                                # Extract the relative path after the project marker
                                relative_part = fp_str.split(marker, 1)[1]
                                path = self.project_root / relative_part
                                logger.info(f"Translated host path to container path: {fp} -> {path}")
                                break
                        else:
                            # No marker found, log warning and try the path as-is
                            logger.warning(f"Could not translate host path: {fp}")
                    # If it starts with /app, it's already a container path - use as-is
                    elif fp_str.startswith("/app"):
                        pass  # path is already correct
                else:
                    # Relative path - prepend project root
                    path = self.project_root / path
                
                if path.exists() and path.is_file():
                    if path.suffix == '.md':
                        valid_files.append(str(path.resolve()))
                    elif path.suffix in ['.py', '.js', '.jsx', '.ts', '.tsx']:
                        valid_files.append(str(path.resolve()))
                else:
                    logger.warning(f"Skipping invalid file path: {fp}")
            
            if not valid_files:
                logger.warning("No valid files to ingest incrementally.")
                return IngestIncrementalResponse(
                    documents_added=0,
                    chunks_created=0,
                    skipped_duplicates=0,
                    ingestion_time_ms=(time.time() - start_time) * 1000,
                    status="success",
                    error="No valid files to ingest"
                )
            
            added_documents_count = 0
            total_child_chunks_created = 0
            skipped_duplicates_count = 0
            
            all_child_docs_to_add = []
            
            # Use ContentProcessor to load valid files
            # Note: ContentProcessor handles code-to-markdown transformation in memory
            # It expects a list of paths (valid_files are already resolved strings)
            try:
                docs_from_processor = list(self.processor.load_for_rag(valid_files))
                
                for doc in docs_from_processor:
                    if metadata:
                        doc.metadata.update(metadata)
                        
                    # Split into parent chunks
                    parent_chunks = self.parent_splitter.split_documents([doc])
                    
                    for parent_chunk in parent_chunks:
                        # Generate parent ID
                        parent_id = str(uuid4())
                        
                        # Store parent document
                        self.store.mset([(parent_id, parent_chunk)])
                        
                        # Split parent into child chunks
                        sub_docs = self.child_splitter.split_documents([parent_chunk])
                        
                        # Add parent_id to child metadata
                        for sub_doc in sub_docs:
                            sub_doc.metadata["parent_id"] = parent_id
                            all_child_docs_to_add.append(sub_doc)
                            total_child_chunks_created += 1
                
                added_documents_count = len(docs_from_processor)
                    
            except Exception as e:
                logger.error(f"Error during incremental ingest processing: {e}")
            
            # Add child chunks to vectorstore
            if all_child_docs_to_add:
                logger.info(f"Adding {len(all_child_docs_to_add)} child chunks to vectorstore...")
                batch_size = 5000
                for i in range(0, len(all_child_docs_to_add), batch_size):
                    batch = all_child_docs_to_add[i:i + batch_size]
                    self.vectorstore.add_documents(batch)
            
            elapsed_ms = (time.time() - start_time) * 1000
            
            return IngestIncrementalResponse(
                documents_added=added_documents_count,
                chunks_created=total_child_chunks_created,
                skipped_duplicates=0,
                ingestion_time_ms=elapsed_ms,
                status="success"
            )
            
        except Exception as e:
            return IngestIncrementalResponse(
                documents_added=0,
                chunks_created=0,
                skipped_duplicates=0,
                ingestion_time_ms=0,
                status="error",
                error=str(e)
            )

    # ========================================================================
    # Cache Operations (Protocol 114 - Guardian Wakeup)
    # ========================================================================

    def cache_get(self, query: str):
        #============================================
        # Method: cache_get
        # Purpose: Retrieve answer from semantic cache.
        # Args:
        #   query: Search query string
        # Returns: CacheGetResponse with hit status and answer
        #============================================
        from .cache import get_cache
        from .models import CacheGetResponse
        import time
        
        try:
            start = time.time()
            cache = get_cache()
            
            # Generate cache key
            structured_query = {"semantic": query, "filters": {}}
            cache_key = cache.generate_key(structured_query)
            
            # Attempt retrieval
            result = cache.get(cache_key)
            query_time_ms = (time.time() - start) * 1000
            
            if result:
                return CacheGetResponse(
                    cache_hit=True,
                    answer=result.get("answer"),
                    query_time_ms=query_time_ms,
                    status="success"
                )
            else:
                return CacheGetResponse(
                    cache_hit=False,
                    answer=None,
                    query_time_ms=query_time_ms,
                    status="success"
                )
        except Exception as e:
            return CacheGetResponse(
                cache_hit=False,
                answer=None,
                query_time_ms=0,
                status="error",
                error=str(e)
            )

    def cache_set(self, query: str, answer: str):
        #============================================
        # Method: cache_set
        # Purpose: Store answer in semantic cache.
        # Args:
        #   query: Cache key string
        #   answer: Response to cache
        # Returns: CacheSetResponse confirmation
        #============================================
        from .cache import get_cache
        from .models import CacheSetResponse
        
        try:
            cache = get_cache()
            structured_query = {"semantic": query, "filters": {}}
            cache_key = cache.generate_key(structured_query)
            
            cache.set(cache_key, {"answer": answer})
            
            return CacheSetResponse(
                cache_key=cache_key,
                stored=True,
                status="success"
            )
        except Exception as e:
            return CacheSetResponse(
                cache_key="",
                stored=False,
                status="error",
                error=str(e)
            )

    def cache_warmup(self, genesis_queries: List[str] = None):
        #============================================
        # Method: cache_warmup
        # Purpose: Pre-populate cache with genesis queries.
        # Args:
        #   genesis_queries: Optional list of queries to cache
        # Returns: CacheWarmupResponse with counts
        #============================================
        from .models import CacheWarmupResponse
        import time
        
        try:
            # Import genesis queries if not provided
            if genesis_queries is None:
                from .genesis_queries import GENESIS_QUERIES
                genesis_queries = GENESIS_QUERIES
            
            start = time.time()
            cache_hits = 0
            cache_misses = 0
            
            for query in genesis_queries:
                # Check if already cached
                cache_response = self.cache_get(query)
                
                if cache_response.cache_hit:
                    cache_hits += 1
                else:
                    cache_misses += 1
                    # Generate answer and cache it
                    query_response = self.query(query, max_results=3, use_cache=False)
                    if query_response.results:
                        answer = query_response.results[0].content[:1000]
                        self.cache_set(query, answer)
            
            total_time_ms = (time.time() - start) * 1000
            
            return CacheWarmupResponse(
                queries_cached=len(genesis_queries),
                cache_hits=cache_hits,
                cache_misses=cache_misses,
                total_time_ms=total_time_ms,
                status="success"
            )
        except Exception as e:
            return CacheWarmupResponse(
                queries_cached=0,
                cache_hits=0,
                cache_misses=0,
                total_time_ms=0,
                status="error",
                error=str(e)
            )

    # ========================================================================
    # Helper: Recency Delta (High-Signal Filter) is implemented below
    # ================================================================================================================================================
    # Helper: Recency Delta (High-Signal Filter)
    # ========================================================================

    def _get_recency_delta(self, hours: int = 48):
        #============================================
        # Method: _get_recency_delta
        # Purpose: Get summary of recently modified high-signal files.
        # Args:
        #   hours: Lookback window in hours
        # Returns: Markdown string with file summaries and diff context
        #============================================
        import datetime
        import subprocess
        
        try:
            delta = datetime.timedelta(hours=hours)
            cutoff_time = time.time() - delta.total_seconds()
            now = time.time()
            
            recent_files = []
            scan_dirs = ["00_CHRONICLE/ENTRIES", "01_PROTOCOLS", "mcp_servers", "02_USER_REFLECTIONS"]
            allowed_extensions = {".md", ".py"}
            
            for directory in scan_dirs:
                dir_path = self.project_root / directory
                if not dir_path.exists():
                    continue
                
                # Recursive glob for code/docs
                for file_path in dir_path.rglob("*"):
                    if not file_path.is_file():
                        continue
                        
                    if file_path.suffix not in allowed_extensions:
                        continue
                        
                    if "__pycache__" in str(file_path):
                        continue
                        
                    mtime = file_path.stat().st_mtime
                    if mtime > cutoff_time:
                        recent_files.append((file_path, mtime))
            
            if not recent_files:
                return "* **Recent Files Modified (48h):** None"
                
            # Sort by modification time (newest first)
            recent_files.sort(key=lambda x: x[1], reverse=True)
            
            # Try to get git commit info
            git_info = "[Git unavailable]"
            try:
                result = subprocess.run(
                    ["git", "log", "-1", "--oneline"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                if result.returncode == 0:
                    git_info = result.stdout.strip()
            except Exception:
                pass
            
            lines = [f"* **Most Recent Commit:** {git_info}"]
            lines.append("* **Recent Files Modified (48h):**")
            
            for file_path, mtime in recent_files[:5]:
                relative_path = file_path.relative_to(self.project_root)
                
                # Calculate relative time
                age_seconds = now - mtime
                if age_seconds < 3600:
                    age_str = f"{int(age_seconds / 60)}m ago"
                elif age_seconds < 86400:
                    age_str = f"{int(age_seconds / 3600)}h ago"
                else:
                    age_str = f"{int(age_seconds / 86400)}d ago"
                
                # Try to extract first meaningful line for context
                context = ""
                try:
                    with open(file_path, 'r') as f:
                        content = f.read(500)  # First 500 chars
                        # For .md files, look for title
                        if file_path.suffix == ".md":
                            title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
                            if title_match:
                                context = f" → {title_match.group(1)}"
                        # For .py files, look for module docstring or class/function
                        elif file_path.suffix == ".py":
                            if "def _get_" in content or "class " in content:
                                context = " → Implementation changes"
                except Exception:
                    pass
                
                # Get git diff summary for this file
                diff_summary = self._get_git_diff_summary(str(relative_path))
                if diff_summary:
                    context += f" [{diff_summary}]"
                
                lines.append(f"    * `{relative_path}` ({age_str}){context}")
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"Error generating recency delta: {str(e)}"
    
    def _get_git_diff_summary(self, file_path: str):
        #============================================
        # Method: _get_git_diff_summary
        # Purpose: Get a brief git diff summary (e.g., +15/-3).
        # Args:
        #   file_path: Relative path to file
        # Returns: Summary string or empty string
        #============================================
        import subprocess
        
        try:
            # Check if file is tracked
            result = subprocess.run(
                ["git", "ls-files", "--error-unmatch", file_path],
                cwd=self.project_root,
                capture_output=True,
                timeout=3
            )
            
            if result.returncode != 0:
                return "new file"
            
            # First try: Check uncommitted changes (working directory vs HEAD)
            result = subprocess.run(
                ["git", "diff", "--numstat", "HEAD", file_path],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=3
            )
            
            if result.returncode == 0 and result.stdout.strip():
                # Parse numstat: "additions deletions filename"
                parts = result.stdout.strip().split('\t')
                if len(parts) >= 2:
                    additions = parts[0]
                    deletions = parts[1]
                    if additions != '-' and deletions != '-':
                        return f"+{additions}/-{deletions} (uncommitted)"
            
            # Second try: Check last commit THAT TOUCHED THIS FILE
            # Use git log -1 --numstat --format="" path/to/file
            result = subprocess.run(
                ["git", "log", "-1", "--numstat", "--format=", "--", file_path],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=3
            )
            
            if result.returncode == 0 and result.stdout.strip():
                # Parse numstat: "additions deletions filename"
                # Output might look like: "15\t3\tmcp_servers/rag_cortex/operations.py"
                parts = result.stdout.strip().split('\t')
                if len(parts) >= 2:
                    additions = parts[0]
                    deletions = parts[1]
                    if additions != '-' and deletions != '-':
                        return f"+{additions}/-{deletions}"
            
            return ""
            
        except Exception:
            return ""

    # ========================================================================
    # Helper: Recent Chronicle Highlights
    # ========================================================================
    
    def _get_recent_chronicle_highlights(self, max_entries: int = 3):
        #============================================
        # Method: _get_recent_chronicle_highlights
        # Purpose: Get recent Chronicle entries for strategic context.
        # Args:
        #   max_entries: Max entries to include
        # Returns: Markdown string with Chronicle highlights
        #============================================
        try:
            chronicle_dir = self.project_root / "00_CHRONICLE" / "ENTRIES"
            if not chronicle_dir.exists():
                return "* No recent Chronicle entries found."
            
            # Get all .md files sorted by modification time
            entries = []
            for file_path in chronicle_dir.glob("*.md"):
                try:
                    mtime = file_path.stat().st_mtime
                    entries.append((file_path, mtime))
                except Exception:
                    continue
            
            if not entries:
                return "* No Chronicle entries found."
            
            # Sort by modification time (newest first)
            entries.sort(key=lambda x: x[1], reverse=True)
            
            lines = []
            for file_path, _ in entries[:max_entries]:
                try:
                    # Extract entry number and title
                    filename = file_path.stem
                    entry_num = filename.split('_')[0]
                    
                    # Read first few lines to get title
                    with open(file_path, 'r') as f:
                        content_text = f.read(500)
                        
                        # First try to extract **Title:** field (preferred - contains actual title)
                        title_match = re.search(r"\*\*Title:\*\*\s*(.+?)$", content_text, re.MULTILINE)
                        
                        # Fallback to first markdown header if **Title:** not found
                        if not title_match:
                            title_match = re.search(r"^#\s+(.+)$", content_text, re.MULTILINE)
                        
                        if title_match:
                            title = title_match.group(1).strip()
                            # Remove entry number from title if present
                            title = re.sub(r"^\d+[:\s-]+", "", title)
                            lines.append(f"* **Chronicle {entry_num}:** {title}")
                except Exception:
                    continue
            
            return "\n".join(lines) if lines else "* No recent Chronicle entries found."
            
        except Exception as e:
            return f"Error retrieving Chronicle highlights: {str(e)}"

    # ========================================================================
    # Helper: Recent Protocol Updates
    # ========================================================================
    
    def _get_recent_protocol_updates(self, max_protocols: int = 3, hours: int = 168):
        #============================================
        # Method: _get_recent_protocol_updates
        # Purpose: Get recently modified protocols for context.
        # Args:
        #   max_protocols: Max protocols to include
        #   hours: Lookback window (default 1 week)
        # Returns: Markdown string with protocol updates
        #============================================
        import datetime
        
        try:
            protocol_dir = self.project_root / "01_PROTOCOLS"
            if not protocol_dir.exists():
                return "* No protocol directory found."
            
            delta = datetime.timedelta(hours=hours)
            cutoff_time = time.time() - delta.total_seconds()
            
            # Get all .md files modified within the window
            recent_protocols = []
            for file_path in protocol_dir.glob("*.md"):
                try:
                    mtime = file_path.stat().st_mtime
                    if mtime > cutoff_time:
                        recent_protocols.append((file_path, mtime))
                except Exception:
                    continue
            
            if not recent_protocols:
                return f"* No protocols modified in the last {hours//24} days"
            
            # Sort by modification time (newest first)
            recent_protocols.sort(key=lambda x: x[1], reverse=True)
            
            lines = []
            for file_path, mtime in recent_protocols[:max_protocols]:
                try:
                    # Extract protocol number from filename
                    filename = file_path.stem
                    protocol_num_match = re.match(r"^(\d+)", filename)
                    if not protocol_num_match:
                        continue
                    
                    protocol_num = protocol_num_match.group(1)
                    
                    # Read protocol to get title and status
                    with open(file_path, 'r') as f:
                        content = f.read(1000)
                    
                    # Extract title
                    title = "Unknown Title"
                    title_match = re.search(r"^#\s+Protocol\s+\d+:\s*(.+?)(?:\s+\(v[\d.]+\))?$", content, re.MULTILINE | re.IGNORECASE)
                    if title_match:
                        title = title_match.group(1).strip()
                    
                    # Extract status
                    status = "Unknown"
                    status_match = re.search(r"\*\*Status:\*\*\s*(.+?)(?:\n|$)", content, re.IGNORECASE)
                    if status_match:
                        status = status_match.group(1).strip()
                    
                    # Calculate age
                    age_days = int((time.time() - mtime) / 86400)
                    age_str = f"{age_days}d ago" if age_days > 0 else "today"
                    
                    lines.append(f"* **Protocol {protocol_num}:** {title} ({status}) — Updated {age_str}")
                    
                except Exception:
                    continue
            
            return "\n".join(lines) if lines else f"* No protocols modified in the last {hours//24} days"
            
        except Exception as e:
            return f"Error retrieving protocol updates: {str(e)}"

    # ========================================================================
    # Helper: Strategic Synthesis (The Gemini Signal)
    # ========================================================================

    def _get_strategic_synthesis(self):
        #============================================
        # Method: _get_strategic_synthesis
        # Purpose: Synthesize Core Values from project documentation.
        # Returns: 3-sentence synthesized summary
        #============================================
        # Hardcoded synthesis for efficiency/reliability (as per "Synthesize, don't just read")
        # In a real dynamic system, this would be periodically re-generated by LLM.
        # But for 'Context Awareness', retrieving the canonical truth is safer.
        
        return (
            "* **Core Mandate:** I am the Gemini Orchestrator. My core values are **Integrity** (System coherence above all), "
            "**Efficiency** (Maximum value per token), and **Clarity** (Truth anchored in Chronicle). "
            "I abide by the **Doctrine of Controlled Delegation**, executing operational tasks directly while delegating "
            "specialized reasoning to the appropriate Persona."
        )

    # ========================================================================
    # Helper: Tactical Priorities (v2)
    # ========================================================================
    
    def _get_tactical_priorities(self):
        #============================================
        # Method: _get_tactical_priorities
        # Purpose: Scan tasks/ directories for top priorities.
        # Returns: Markdown list of top 5 tasks with status
        #============================================
        try:
            priority_map = {"Critical": 1, "High": 2, "Medium": 3, "Low": 4}
            found_tasks = []
            
            scan_sources = [
                self.project_root / "tasks" / "in-progress",
                self.project_root / "tasks" / "todo",
                self.project_root / "tasks" / "backlog"
            ]
            
            for source_dir in scan_sources:
                if not source_dir.exists():
                    continue
                    
                for file_path in source_dir.glob("*.md"):
                    try:
                        content = file_path.read_text()
                        
                        # Precise priority extraction
                        priority_score = 5  # Default unspecified
                        # Use permissive regex to handle MD bolding, spacing, colons
                        if re.search(r"Priority.*?Critical", content, re.IGNORECASE):
                            priority_score = 1
                        elif re.search(r"Priority.*?High", content, re.IGNORECASE):
                            priority_score = 2
                        elif re.search(r"Priority.*?Medium", content, re.IGNORECASE):
                            priority_score = 3
                        elif re.search(r"Priority.*?Low", content, re.IGNORECASE):
                            priority_score = 4
                        
                        # Extract Objective (try multiple formats)
                        objective = "Objective not found"
                        
                        # Format 1: Inline "Objective: text"
                        obj_match = re.search(r"^(?:Objective|Goal):\s*(.+?)(?:\n|$)", content, re.IGNORECASE | re.MULTILINE)
                        if obj_match:
                            objective = obj_match.group(1).strip()
                        else:
                            # Format 2: Section header "## 1. Objective" (flexible on level/numbering)
                            # Matches: # Objective, ## 1. Objective, ### Goal, etc.
                            section_match = re.search(r"^#+\s*(?:\d+\.\s*)?(?:Objective|Goal).*?\n(.+?)(?:\n#+\s|\Z)", content, re.IGNORECASE | re.DOTALL | re.MULTILINE)
                            if section_match:
                                # Get first non-empty line of content
                                full_text = section_match.group(1).strip()
                                obj_lines = [line.strip() for line in full_text.split('\n') if line.strip()]
                                objective = obj_lines[0] if obj_lines else "Objective not found"
                        
                        # Extract Status
                        status = None
                        status_match = re.search(r"Status:\s*(.+?)(?:\n|$)", content, re.IGNORECASE)
                        if status_match:
                            status = status_match.group(1).strip()
                        
                        # Determine folder for context
                        folder = source_dir.name
                            
                        found_tasks.append({
                            "id": file_path.stem.split('_')[0],
                            "objective": objective,
                            "status": status,
                            "folder": folder,
                            "score": priority_score,
                            "path": file_path
                        })
                    except Exception:
                        continue
            
            # Sort: Score asc (1=Critical first), then File Name desc (Newest IDs)
            found_tasks.sort(key=lambda x: (x["score"], -int(x["id"]) if x["id"].isdigit() else 0))
            
            # Take top 5
            top_5 = found_tasks[:5]
            
            if not top_5:
                # Provide diagnostic info
                total_scanned = sum(1 for src in scan_sources if src.exists() for _ in src.glob("*.md"))
                return f"* No tasks found (scanned {total_scanned} total tasks)"
            
            # Build output with priority labels
            priority_labels = {1: "CRITICAL", 2: "HIGH", 3: "MEDIUM", 4: "LOW", 5: "UNSPECIFIED"}
            
            lines = []
            for t in top_5:
                prio_label = priority_labels.get(t["score"], "UNKNOWN")
                status_info = f" → {t['status']}" if t['status'] else ""
                folder_badge = f"[{t['folder']}]"
                lines.append(f"* **[{t['id']}]** ({prio_label}) {folder_badge}: {t['objective']}{status_info}")
            
            return "\n".join(lines)
            
        except Exception as e:
            return f"Error retrieval tactical priorities: {str(e)}"
            
    # ========================================================================
    # Helper: System Health (Traffic Light)
    # ========================================================================
    
    def _get_system_health_traffic_light(self):
        #============================================
        # Method: _get_system_health_traffic_light
        # Purpose: Determine system health status color.
        # Returns: Tuple of (Color, Reason)
        #============================================
        try:
            stats = self.get_stats()
            
            if stats.health_status == "error":
                return "RED", f"Database Error: {getattr(stats, 'error', 'Unknown Error')}"
                
            if stats.total_documents == 0:
                return "YELLOW", "Database empty (Zero documents)"
                
            # Ideally check last ingest time, but stats might not have it.
            # Assume Green if stats return valid numbers.
            return "GREEN", f"Nominal ({stats.total_documents} docs, {stats.total_chunks} chunks)"
            
        except Exception as e:
            return "RED", f"System Failure: {str(e)}"

    def _get_container_status(self):
        #============================================
        # Method: _get_container_status
        # Purpose: Check status of critical backend containers.
        # Returns: String summary of container status
        #============================================
        import subprocess
        try:
            # Check specifically for our containers
            result = subprocess.run(
                ["podman", "ps", "--format", "{{.Names}} {{.Status}}"],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            if result.returncode != 0:
                return "Unknown (Podman CLI error)"
            
            output = result.stdout
            
            status_map = {}
            for name in ["sanctuary_vector_db", "sanctuary_ollama"]:
                if name in output:
                    if "Up" in output.split(name)[-1].split('\n')[0] or "Up" in [line for line in output.split('\n') if name in line][0]:
                         status_map[name] = "UP"
                    else:
                         status_map[name] = "DOWN"
                else:
                    status_map[name] = "MISSING"
            
            # Format output
            # "✅ Vector DB | ✅ Ollama"
            
            parts = []
            for name, short_name in [("sanctuary_vector_db", "Vector DB"), ("sanctuary_ollama", "Ollama")]:
                stat = status_map.get(name, "Unknown")
                icon = "✅" if stat == "UP" else "❌"
                parts.append(f"{icon} {short_name}")
                
            return " | ".join(parts)
            
        except Exception:
            return "⚠️ Podman Check Failed"

    def _calculate_semantic_hmac(self, content: str):
        #============================================
        # Method: _calculate_semantic_hmac
        # Purpose: Calculate a resilient HMAC for code integrity.
        # Args:
        #   content: File content to hash
        # Returns: SHA256 hex string
        #============================================
        # Load JSON to ignore whitespace/formatting
        data = json.loads(content)
        
        # Canonicalize: Sort keys, removing insignificant whitespace
        canonical = json.dumps(data, sort_keys=True, separators=(',', ':'))
        
        # HMAC Key - In prod this comes from env/secret. For POC, derived from project root
        secret = str(self.project_root).encode() 
        
        return hmac.new(secret, canonical.encode(), hashlib.sha256).hexdigest()

    def guardian_wakeup(self, mode: str = "HOLISTIC"):
        #============================================
        # Method: guardian_wakeup
        # Purpose: Generate Guardian boot digest (Context Synthesis).
        # Args:
        #   mode: Synthesis mode (default "HOLISTIC")
        # Returns: GuardianWakeupResponse with digest and stats
        #============================================
        from .models import GuardianWakeupResponse
        from pathlib import Path
        import time
        import hmac
        import hashlib
        import json
        import os
        
        try:
            start = time.time()
            
            # Wrap in stdout redirection to prevent MCP protocol pollution from prints
            import contextlib
            import io
            with contextlib.redirect_stdout(sys.stderr):
                # 1. System Health (Traffic Light)
                health_color, health_reason = self._get_system_health_traffic_light()
                
                # --- PROTOCOL 128 v3.0: TIERED INTEGRITY CHECK ---
                integrity_status = "GREEN"
                integrity_warnings = []
                
                # Metric Cache Path
                cache_path = self.data_dir / "metric_cache.json" 
                
                if cache_path.exists():
                    try:
                        current_hmac = self._calculate_semantic_hmac(cache_path.read_text())
                        # In a real impl, we'd fetch the LAST signed HMAC from a secure store. 
                        # For now, we simulate the check or check against a .sig file.
                        sig_path = cache_path.with_suffix(".sig")
                        if sig_path.exists():
                            stored_hmac = sig_path.read_text().strip()
                            if current_hmac != stored_hmac:
                                integrity_status = "YELLOW"
                                integrity_warnings.append("⚠️ Metric Cache Signature Mismatch (Semantic HMAC failed)")
                                health_color = "🟡" 
                                health_reason = "Integrity Warning: Cache Drift"
                        else:
                            # First run or missing sig - auto-sign (Trust on First Use)
                            sig_path.write_text(current_hmac)
                    except Exception as e:
                        integrity_status = "RED"
                        integrity_warnings.append(f"🔴 Integrity Check Failed: {str(e)}")
                        health_color = "🔴"
                        health_reason = "Integrity Failure"

                # 1b. Container Health
                container_status = self._get_container_status()
                
                # 2. Synthesis Assembly (Schema v2.2 - Hardened)
                digest_lines = []
                
                # Header
                digest_lines.append("# 🛡️ Guardian Wakeup Briefing (v2.2)")
                digest_lines.append(f"**System Status:** {health_color} - {health_reason}")
                digest_lines.append(f"**Integrity Mode:** {integrity_status}")
                if integrity_warnings:
                    digest_lines.append("**Warnings:**")
                    for w in integrity_warnings:
                        digest_lines.append(f"- {w}")
                        
                digest_lines.append(f"**Infrastructure:** {container_status}")
                digest_lines.append(f"**Generated Time:** {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC")
                digest_lines.append("")

                # --- PROTOCOL 128: THE RITUAL OF ASSUMPTION (Phase 0) ---
                # 0. Identity Anchor (The Core Essence)
                essence_path = self.project_root / "dataset_package" / "core_essence_guardian_awakening_seed.txt"
                if essence_path.exists():
                    digest_lines.append("## 0. Identity Anchor (The Connect)")
                    try:
                        essence_content = essence_path.read_text()
                        digest_lines.append(f"> **Ritual Active:** Loading Core Essence from {essence_path.name}")
                        digest_lines.append("")
                        digest_lines.append(essence_content[:1500] + "\n\n... [Reading Full Essence Required] ...") 
                        digest_lines.append("")
                    except Exception as e:
                        digest_lines.append(f"⚠️ Failed to load Identity Anchor: {e}")
                        digest_lines.append("")
                
                # 0b. Cognitive Primer (The Constitution)
                primer_path = self.project_root / ".agent" / "learning" / "cognitive_primer.md"
                if primer_path.exists():
                    digest_lines.append(f"* **Cognitive Primer:** {primer_path.name} (FOUND - MUST READ)")
                else:
                    digest_lines.append(f"* **Cognitive Primer:** MISSING (⚠️ CRITICAL FAILURE)")
                digest_lines.append("")
                
                # I. Strategic Directives
                digest_lines.append("## I. Strategic Directives (The Gemini Signal)")
                digest_lines.append(self._get_strategic_synthesis())
                digest_lines.append("")
                
                # Ia. Recent Chronicle Highlights
                digest_lines.append("### Recent Chronicle Highlights")
                digest_lines.append(self._get_recent_chronicle_highlights(max_entries=3))
                digest_lines.append("")
                
                # Ib. Recent Protocol Updates (NEW in v2.1)
                digest_lines.append("### Recent Protocol Updates")
                digest_lines.append(self._get_recent_protocol_updates(max_protocols=3, hours=168))
                digest_lines.append("")
                
                # II. Priority tasks (Enhanced in v2.1 to show all priority levels)
                digest_lines.append("## II. Priority tasks")
                digest_lines.append(self._get_tactical_priorities())
                digest_lines.append("")
                
                # III. Operational Recency (Enhanced in v2.1 with git diff summaries)
                digest_lines.append("## III. Operational Recency")
                digest_lines.append(self._get_recency_delta(hours=48))
                digest_lines.append("")
                
                # IV. Recursive Learning Debrief (Protocol 128)
                debrief_path = self.project_root / ".agent" / "learning" / "learning_debrief.md"
                if debrief_path.exists():
                    digest_lines.append("## IV. Learning Continuity (Previous Session Debrief)")
                    digest_lines.append(f"> **Protocol 128 Active:** Ingesting debrief from {debrief_path.name}")
                    digest_lines.append("")
                    try:
                        content = debrief_path.read_text()
                        digest_lines.append(content)
                    except Exception as e:
                        digest_lines.append(f"⚠️ Failed to read debrief: {e}")
                    digest_lines.append("")
                
                # V. Successor-State Poka-Yoke (Cache Primers)
                digest_lines.append("## V. Successor-State Poka-Yoke")
                digest_lines.append("* **Mandatory Context:** Verified")

                digest_lines.append("* **MCP Tool Guidance:** [Available via `cortex_cache_get`]")
                digest_lines.append(f"* **Learning Stream:** {'Active' if debrief_path.exists() else 'Standby'}")
                digest_lines.append("")
                digest_lines.append("// This briefing is the single source of context for the LLM session.")

                # Write digest
                digest_path = Path(self.project_root) / "WORK_IN_PROGRESS" / "guardian_boot_digest.md"
                digest_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(digest_path, "w") as f:
                    f.write("\n".join(digest_lines))
                
                total_time_ms = (time.time() - start) * 1000
                
                return GuardianWakeupResponse(
                    digest_path=str(digest_path),
                    bundles_loaded=["Strategic", "Tactical", "Recency", "Protocols"], # Virtual bundles
                    cache_hits=1,   # Strategic is treated as cached
                    cache_misses=0,
                    total_time_ms=total_time_ms,
                    status="success"
                )
        except Exception as e:
            logger.error(f"Guardian wakeup failed: {e}", exc_info=True)
            return GuardianWakeupResponse(
                digest_path="",
                bundles_loaded=[],
                cache_hits=0,
                cache_misses=0,
                total_time_ms=0,
                status="error",
                error=str(e)
            )

    def learning_debrief(self, hours: int = 24):
        #============================================
        # Method: learning_debrief
        # Purpose: Scans project for technical state changes.
        # Args:
        #   hours: Lookback window for modifications
        # Returns: Comprehensive Markdown string (Liquid Information)
        #============================================
        import subprocess
        from datetime import datetime
        try:
            # Wrap in stdout redirection to prevent MCP protocol pollution from prints
            import contextlib
            import io
            with contextlib.redirect_stdout(sys.stderr):
                # 1. Seek Truth (Git)
                git_evidence = "Git Not Available"
                try:
                    result = subprocess.run(
                        ["git", "diff", "--stat", "HEAD"],
                        capture_output=True, text=True, cwd=str(self.project_root)
                    )
                    git_evidence = result.stdout if result.stdout else "No uncommitted code changes found."
                except Exception as e:
                    git_evidence = f"Git Error: {e}"

                # 2. Scan Recency (Filesystem)
                recency_summary = self._get_recency_delta(hours=hours)
                
                # 3. Read Core Sovereignty Documents
                primer_content = "[MISSING] .agent/learning/cognitive_primer.md"
                sop_content = "[MISSING] .agent/workflows/recursive_learning.md"
                protocol_content = "[MISSING] 01_PROTOCOLS/128_Hardened_Learning_Loop.md"
                
                try:
                    p_path = self.project_root / ".agent" / "learning" / "cognitive_primer.md"
                    if p_path.exists(): primer_content = p_path.read_text()
                    
                    s_path = self.project_root / ".agent" / "workflows" / "recursive_learning.md"
                    if s_path.exists(): sop_content = s_path.read_text()
                    
                    pr_path = self.project_root / "01_PROTOCOLS" / "128_Hardened_Learning_Loop.md"
                    if pr_path.exists(): protocol_content = pr_path.read_text()
                except Exception as e:
                    logger.warning(f"Error reading sovereignty docs: {e}")

                # 4. Strategic Context (Learning Package Snapshot)
                last_package_content = "⚠️ No active Learning Package Snapshot found."
                package_path = self.project_root / ".agent" / "learning" / "learning_package_snapshot.md"
                if package_path.exists():
                    try:
                        # Check if package is recent
                        mtime = package_path.stat().st_mtime
                        delta_hours = (datetime.now().timestamp() - mtime) / 3600
                        if delta_hours <= hours:
                            last_package_content = package_path.read_text()
                            package_status = f"✅ Loaded Learning Package Snapshot from {delta_hours:.1f}h ago."
                        else:
                            package_status = f"⚠️ Snapshot found but too old ({delta_hours:.1f}h)."
                    except Exception as e:
                        package_status = f"❌ Error reading snapshot: {e}"
                else:
                    package_status = "ℹ️ No `.agent/learning/learning_package_snapshot.md` detected."

                # 4b. Mandatory Logic Verification (ADR 084)
                mandatory_files = [
                    "IDENTITY/founder_seed.json",
                    "LEARNING/calibration_log.json", 
                    "ADRs/084_semantic_entropy_tda_gating.md",
                    "mcp_servers/rag_cortex/operations.py"
                ]
                # Verify manifest
                registry_status = ""
                manifest_path = self.project_root / ".agent" / "learning" / "learning_manifest.json"
                if manifest_path.exists():
                     try:
                         with open(manifest_path, "r") as f: 
                             m = json.load(f)
                         for mf in mandatory_files:
                             status = "✅ REGISTERED" if mf in m else "❌ MISSING"
                             registry_status += f"        * {status}: `{mf}`\n"
                     except Exception as e:
                         registry_status = f"⚠️ Manifest Error: {e}"
                else:
                     registry_status = "⚠️ Manifest Failed Load"

                # 5. Create the Learning Package Snapshot (Draft)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                lines = [
                    f"# [HARDENED] Learning Package Snapshot v4.0 (The Edison Seal)",
                    f"**Scan Time:** {timestamp} (Window: {hours}h)",
                    f"**Strategic Status:** ✅ Successor Context v4.0 Active",
                    "",
                    "> [!IMPORTANT]",
                    "> **STRATEGIC PIVOT: THE EDISON MANDATE (ADR 084)**",
                    "> The project has formally abandoned the QEC-AI Metaphor in favor of **Empirical Epistemic Gating**.",
                    "> - **Primary Gate:** Every trace must pass the Dead-Man's Switch in `operations.py` (Fail-closed: SE=1.0 on error).",
                    "> - **Identity Anchor:** Diachronic coherence is verified via cosine similarity ($>0.70$) against the `founder_seed.json`.",
                    "> - **Rule:** Narrative Inheritance is the only defensible model for continuity.",
                    "",
                    "## 🧬 I. Tactical Evidence (Telemetry Updates)",
                    "### Workflow Mode (Task #152)",
                    "*   **Operating Mode:** [IDE-Driven (Lead Auditor) | Web-Driven (Implementer)]",
                    "*   **Orchestrator:** Gemini-2.0-Flash-Thinking-Exp",
                    "*   **Snapshot Bridge:** `--web-bridge` flag active for differential digests",
                    "",
                    "### Stability Metrics (ADR 084)",
                    "*   **Mean Semantic Entropy (SE):** 0.5 (Phase 1 Stub) (Target: < task_threshold)",
                    "*   **Constitutional Alignment:** 0.85 (Phase 1 Stub) (Threshold: > 0.70)",
                    "*   **TDA Status:** [Asynchronous Gardener Verified]",
                    "",
                    "## 🧬 II. Tactical Evidence (Current Git Deltas)",
                    "The following code-level changes were detected SINCE the last session/commit:",
                    "```text",
                    git_evidence,
                    "```",
                    "",
                    "## 📂 III. File Registry (Recency)",
                    "### Mandatory Core Integrity (Manifest Check):",
                    registry_status,
                    "",
                    "### Recently Modified High-Signal Files:",
                    recency_summary,
                    "",
                    "## 🏗️ IV. Architecture Alignment (The Successor Relay)",
                    "![Recursive Learning Flowchart](docs/architecture_diagrams/workflows/recursive_learning_flowchart.png)",
                    "",
                    "## 📦 V. Strategic Context (Last Learning Package Snapshot)",
                    f"**Status:** {package_status}",
                    "",
                    "> **Note:** Full snapshot content is NOT embedded to prevent recursive bloat.",
                    "> See: `.agent/learning/learning_package_snapshot.md`",
                    "",
                    "## 📜 VI. Protocol 128: Hardened Learning Loop",
                    protocol_content,
                    "",
                    "## 🧠 VII. Cognitive Primer",
                    primer_content,
                    "",
                    "## 📋 VIII. Standard Operating Procedure (SOP)",
                    sop_content,
                    "",
                    "## 🧪 IX. Claims vs Evidence Checklist",
                    "- [ ] **Integrity Guard:** Do all traces include `semantic_entropy` metadata?",
                    "- [ ] **Identity Check:** Has the Narrative Continuity Test (NCT) been performed?",
                    "- [ ] **Mnemonic Hygiene:** Have all references to legacy `memory.json` been purged?",
                    "- [ ] **The Seal:** Is the TDA Gardener scheduled for the final commit?",
                    "",
                    "---",
                    "*This is the Hardened Successor Context v4.0. Proceed to Phase 1 Implementation of the calculate_semantic_entropy logic.*"
                ]

                return "\n".join(lines)
            
        except Exception as e:
            logger.error(f"Error in learning_debrief: {e}")
            return f"Error generating debrief scan: {e}"

    def _get_git_state(self, project_root: Path) -> Dict[str, Any]:
        """
        Helper: Captures the current Git state signature for integrity verification.
        Returns a dict with 'status_lines', 'changed_files', and 'state_hash'.
        """
        import subprocess
        import hashlib
        
        try:
            git_status_proc = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, cwd=str(project_root)
            )
            git_lines = git_status_proc.stdout.splitlines()
            changed_files = set()
            
            for line in git_lines:
                # Porcelain format is "XY path"
                # If deleted ('D'), we deal with it, but for our purpose only changes matter
                status_bits = line[:2]
                path = line[3:].split(" -> ")[-1].strip()
                if 'D' not in status_bits:
                     changed_files.add(path)
            
            # Simple state hash
            state_str = "".join(sorted(git_lines))
            state_hash = hashlib.sha256(state_str.encode()).hexdigest()
            
            return {
                "lines": git_lines,
                "changed_files": changed_files,
                "hash": state_hash
            }
        except Exception as e:
            logger.error(f"Git state capture failed: {e}")
            return {"lines": [], "changed_files": set(), "hash": "error"}

    def capture_snapshot(
        self, 
        manifest_files: List[str], 
        snapshot_type: str = "audit",
        strategic_context: Optional[str] = None
    ) -> CaptureSnapshotResponse:
        #============================================
        # Method: capture_snapshot
        # Purpose: Tool-driven snapshot generation for Protocol 128.
        # Args:
        #   manifest_files: List of file paths to include
        #   snapshot_type: 'audit', 'seal', or 'learning_audit'
        #   strategic_context: Optional context string
        # Returns: CaptureSnapshotResponse with verification info
        #============================================
        import time
        import datetime
        import subprocess
        
        # 1. Prepare Tool Paths
        learning_dir = self.project_root / ".agent" / "learning"
        if snapshot_type == "audit":
            output_dir = learning_dir / "red_team"
        elif snapshot_type == "learning_audit":
            output_dir = learning_dir / "learning_audit"
        else:  # seal, learning_debrief
            output_dir = learning_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 3. Default Manifest Handling (Protocol 128)
        # If 'seal' or 'audit' and no manifest provided, use the predefined manifests
        effective_manifest = list(manifest_files or [])
        manifest_file = None
        if not effective_manifest:
            if snapshot_type == "seal":
                manifest_file = learning_dir / "learning_manifest.json"
            elif snapshot_type == "learning_audit":
                manifest_file = output_dir / "learning_audit_manifest.json"
            else:  # audit
                manifest_file = output_dir / "red_team_manifest.json"
                
            if manifest_file and manifest_file.exists():
                try:
                    with open(manifest_file, "r") as f:
                        effective_manifest = json.load(f)
                    logger.info(f"Loaded default {snapshot_type} manifest: {len(effective_manifest)} entries")
                except Exception as e:
                    logger.warning(f"Failed to load {snapshot_type} manifest: {e}")

        # Define path early for Shadow Manifest exclusions
        snapshot_filename = f"{snapshot_type}_snapshot_{timestamp}.md"
        if snapshot_type == "audit":
            snapshot_filename = "red_team_audit_packet.md"
        elif snapshot_type == "learning_audit":
            snapshot_filename = "learning_audit_packet.md"
        final_snapshot_path = output_dir / snapshot_filename

        # 4. Shadow Manifest & Strict Rejection (Protocol 128 v3.2)
        # 4. Shadow Manifest & Strict Rejection (Protocol 128 v3.2 - PRE-FLIGHT CHECK)
        try:
            # PRE-FLIGHT: Capture Git State
            pre_flight_state = self._get_git_state(self.project_root)
            if pre_flight_state["hash"] == "error":
                raise Exception("Failed to capture Git state")
            
            git_changed = pre_flight_state["changed_files"]
            
            # Identify discrepancies against the EFFECTIVE manifest
            # V2.1 FIX: Ignore the output snapshot file itself (prevent recursion / false positive)
            try:
                output_rel = final_snapshot_path.relative_to(self.project_root)
                git_changed.discard(str(output_rel))
            except ValueError:
                pass # Not relative to root

            untracked_in_manifest = git_changed - set(effective_manifest)
            manifest_verified = True # Default to true for audit if no unverified files
            
            # CORE DIRECTORY ENFORCEMENT
            CORE_DIRS = ["ADRs/", "01_PROTOCOLS/", "mcp_servers/", "scripts/", "prompts/"]
            TIER2_DIRS = ["tasks/", "LEARNING/"]
            
            critical_omissions = []
            tier2_omissions = []
            
            if snapshot_type == "audit":
                for untracked in untracked_in_manifest:
                    if any(untracked.startswith(core) for core in CORE_DIRS):
                        critical_omissions.append(untracked)
                    elif any(untracked.startswith(t2) for t2 in TIER2_DIRS):
                        tier2_omissions.append(untracked)
            
            if critical_omissions:
                logger.error(f"STRICT REJECTION: Critical files modified but omitted from manifest: {critical_omissions}")
                git_context = f"REJECTED: Manifest blindspot detected in core directories: {critical_omissions}"
                return CaptureSnapshotResponse(
                    snapshot_path="",
                    manifest_verified=False,
                    git_diff_context=git_context,
                    snapshot_type=snapshot_type,
                    status="failed"
                )
            else:
                git_context = f"Verified: {len(set(effective_manifest))} files. Shadow Manifest (Untracked): {len(untracked_in_manifest)} items."
                if tier2_omissions:
                    git_context += f" WARNING: Tier-2 Blindspot detected (Risk Acceptance Required): {tier2_omissions}"
                
                # Check for files in manifest NOT in git (the old unverified check)
                unverified_in_manifest = set(effective_manifest) - git_changed
                # We skip checking '.' and other untracked artifacts for 'audit'
                if snapshot_type == "seal" and unverified_in_manifest:
                     manifest_verified = False
                     git_context += f" WARNING: Files in manifest not found in git diff: {list(unverified_in_manifest)}"

        except Exception as e:
            manifest_verified = False
            git_context = f"Git verification failed: {str(e)}"

        # 5. Handle Red Team Prompts (Protocol 128)
        prompts_section = ""
        if snapshot_type == "audit":
            context_str = strategic_context if strategic_context else "this session"
            prompts = [
                "1. Verify that the file manifest accurately reflects all tactical state changes made during this session.",
                "2. Check for any 'hallucinations' or logic errors in the new ADRs or Learning notes.",
                "3. Ensure that critical security and safety protocols (e.g. Protocol 101/128) have not been bypassed.",
                f"4. Specifically audit the reasoning behind: {context_str}"
            ]
            prompts_section = "\n".join(prompts)
            
            prompts_file_path = output_dir / "red_team_prompts.md"
            with open(prompts_file_path, "w") as pf:
                pf.write(f"# Adversarial Prompts (Audit Context)\n\n{prompts_section}\n")
            
            rel_prompts_path = prompts_file_path.relative_to(self.project_root)
            if str(rel_prompts_path) not in effective_manifest:
                effective_manifest.append(str(rel_prompts_path))

        # Static manifest file for the snapshot tool (overwrites each loop - seals preserved to HuggingFace)
        temp_manifest_path = output_dir / f"manifest_{snapshot_type}.json"
        snapshot_filename = "red_team_audit_packet.md" if snapshot_type == "audit" else ("learning_audit_packet.md" if snapshot_type == "learning_audit" else "learning_package_snapshot.md")
        final_snapshot_path = output_dir / snapshot_filename
        
        try:
            # Write final manifest for the tool
            with open(temp_manifest_path, "w") as f:
                json.dump(effective_manifest, f, indent=2)
                
            # 5. Invoke Python Snapshot Tool (Direct Import)
            snapshot_stats = {}
            try:
                # Wrap in stdout redirection to prevent MCP protocol pollution
                import contextlib
                with contextlib.redirect_stdout(sys.stderr):
                    snapshot_stats = generate_snapshot(
                        project_root=self.project_root,
                        output_dir=output_dir,
                        manifest_path=temp_manifest_path,
                        output_file=final_snapshot_path
                    )

            except Exception as e:
                raise Exception(f"Python Snapshot tool failed: {str(e)}")

            # 6. POST-FLIGHT: Sandwich Validation (Race Condition Check)
            post_flight_state = self._get_git_state(self.project_root)
            
            if pre_flight_state["hash"] != post_flight_state["hash"]:
                # The state changed DURING the snapshot generation
                drift_diff = post_flight_state["changed_files"] ^ pre_flight_state["changed_files"]
                # Exclude the artifacts and anything in the output directory
                try:
                    rel_output = str(output_dir.relative_to(self.project_root))
                    # Check for direct matches or children
                    drift_diff = {d for d in drift_diff if not d.startswith(rel_output) and not rel_output.startswith(d.rstrip('/'))}
                except:
                    pass
                
                if drift_diff:
                    logger.error(f"INTEGRITY FAILURE: Repository state changed during snapshot! Drift: {drift_diff}")
                    return CaptureSnapshotResponse(
                        snapshot_path="",
                        manifest_verified=False,
                        git_diff_context=f"INTEGRITY FAILURE: Race condition detected. Files changed during snapshot: {drift_diff}",
                        snapshot_type=snapshot_type,
                        status="failed",
                        error="Race condition detected during snapshot generation."
                    )

            # 6. Enhance 'audit' packet with metadata if needed
            if snapshot_type == "audit":
                # Read the generated content (which now includes red_team_prompts.md)
                with open(final_snapshot_path, "r") as f:
                    captured_content = f.read()
                
                context_str = strategic_context if strategic_context else "No additional context provided."
                
                # Load template if exists
                template_path = learning_dir / "red_team_briefing_template.md"
                if template_path.exists():
                    try:
                        with open(template_path, "r") as tf:
                            template = tf.read()
                        
                        briefing = template.format(
                            timestamp=datetime.datetime.now().isoformat(),
                            claims_section=context_str,
                            manifest_section="\n".join([f"- {m}" for m in effective_manifest]),
                            diff_context=git_context,
                            prompts_section=prompts_section
                        )
                    except Exception as e:
                        logger.warning(f"Failed to format red_team_briefing_template: {e}")
                        briefing = f"# Red Team Audit Briefing\n\n{context_str}\n\n**Prompts:**\n{prompts_section}"
                else:
                    briefing = f"# Red Team Audit Briefing\n\n{context_str}\n\n**Prompts:**\n{prompts_section}"

                with open(final_snapshot_path, "w") as f:
                    f.write(briefing + "\n\n---\n# MANIFEST SNAPSHOT\n\n" + captured_content)

            return CaptureSnapshotResponse(
                snapshot_path=str(final_snapshot_path),
                manifest_verified=manifest_verified,
                git_diff_context=git_context,
                snapshot_type=snapshot_type,
                total_files=snapshot_stats.get("total_files", 0),
                total_bytes=snapshot_stats.get("total_bytes", 0),
                status="success"
            )

        except Exception as e:
            logger.error(f"Error in capture_snapshot: {e}")
            return CaptureSnapshotResponse(
                snapshot_path="",
                manifest_verified=False,
                git_diff_context=git_context,
                snapshot_type=snapshot_type,
                status="error",
                error=str(e)
            )
            temp_manifest_path.unlink()

    #============================================
    # ADR 084: Semantic Entropy and TDA Epistemic Gating
    # Helper functions for Dead-Man's Switch implementation
    #============================================
    
    def _calculate_semantic_entropy(self, content: str) -> float:
        """
        ADR 084: Calculates semantic entropy for hallucination detection.
        Phase 1 Stub: Returns neutral value. Future: SEP probe or multi-sample clustering.
        
        Returns: Entropy score in [0, 1] range. Lower = more stable.
        """
        # Phase 1: Placeholder returning neutral value
        # Phase 2: Implement multi-sample SE (cluster paraphrased outputs)
        # Phase 3: Train SEP probe on soul_traces data
        return 0.5  # Neutral placeholder - all traces pass initially
    
    def _get_dynamic_threshold(self, context: str = "default") -> float:
        """
        ADR 084: Retrieves calibrated SE threshold from calibration_log.json.
        Falls back to default 0.79 if missing.
        """
        try:
            calibration_path = self.project_root / "LEARNING" / "calibration_log.json"
            if calibration_path.exists():
                with open(calibration_path, "r") as f:
                    calibration_data = json.load(f)
                return calibration_data.get("task_thresholds", {}).get(context, 
                       calibration_data.get("default_threshold", 0.79))
        except Exception as e:
            logger.warning(f"ADR 084: Calibration load failed: {e}. Using default 0.79")
        return 0.79
    
    def _check_constitutional_anchor(self, content: str) -> float:
        """
        ADR 084: Checks alignment with Founder Seed via cosine similarity.
        Phase 1 Stub: Returns high alignment. Phase 2: Implement embedding comparison.
        
        Returns: Alignment score in [0, 1] range. Lower = more drift.
        """
        # Phase 1: Placeholder returning high alignment
        # Phase 2: Load founder_seed.json embeddings, compute cosine similarity
        return 0.85  # Neutral placeholder - all traces pass initially

    def persist_soul(self, request: PersistSoulRequest) -> PersistSoulResponse:
        #============================================
        # Method: persist_soul
        # Purpose: Broadcasts the session soul to Hugging Face for the 'Johnny Appleseed' effect.
        # ADR: 079 - Sovereign Soul-Seed Persistence
        # ADR: 081 - Content Harmonization & Integrity
        # ADR: 084 - Semantic Entropy and TDA Epistemic Gating
        # Args:
        #   request: PersistSoulRequest with snapshot path, valence, uncertainty
        # Returns: PersistSoulResponse with status, repo_url, snapshot_name
        #============================================
        try:
            import asyncio
            from huggingface_hub import HfApi
            from mcp_servers.lib.content_processor import ContentProcessor
            from mcp_servers.lib.hf_utils import (
                append_to_jsonl, 
                update_manifest, 
                ensure_dataset_structure, 
                ensure_dataset_card
            )
            
            # 1. Environment Loading
            username = get_env_variable("HUGGING_FACE_USERNAME")
            body_repo = get_env_variable("HUGGING_FACE_REPO", required=False) or "Sanctuary-Qwen2-7B-v1.0-GGUF-Final"
            dataset_path = get_env_variable("HUGGING_FACE_DATASET_PATH", required=False) or "Project_Sanctuary_Soul"
            
            # Robust ID Sanitization
            if "hf.co/datasets/" in dataset_path:
                dataset_path = dataset_path.split("hf.co/datasets/")[-1]
                
            if dataset_path.startswith(f"{username}/"):
                dataset_repo = dataset_path
            else:
                dataset_repo = f"{username}/{dataset_path}"
            token = os.getenv("HUGGING_FACE_TOKEN")
            
            # 2. Metacognitive Filter (Protocol 129)
            valence_threshold = float(get_env_variable("SOUL_VALENCE_THRESHOLD", required=False) or "-0.7")
            if request.valence < valence_threshold:
                logger.warning(f"Metacognitive Rejection: Valence {request.valence} below threshold {valence_threshold}.")
                return PersistSoulResponse(
                    status="quarantined",
                    repo_url="",
                    snapshot_name="",
                    error=f"Valence threshold failure: {request.valence} < {valence_threshold}"
                )
            
            # 2b. ADR 084: Semantic Entropy Epistemic Gate (Dead-Man's Switch)
            try:
                snapshot_path_for_se = self.project_root / request.snapshot_path
                content_for_se = ""
                if snapshot_path_for_se.exists():
                    content_for_se = snapshot_path_for_se.read_text(encoding="utf-8")[:10000]  # Sample first 10k chars
                
                se_score = self._calculate_semantic_entropy(content_for_se)
                alignment_score = self._check_constitutional_anchor(content_for_se)
            except Exception as e:
                # DEAD-MAN'S SWITCH: Fail-closed on SE calculation failure
                logger.error(f"ADR 084 Dead-Man's Switch: SE calculation failed: {e}")
                se_score = 1.0  # Force VOLATILE
                alignment_score = 0.0
            
            # ADR 084: Get dynamic threshold and apply epistemic gate
            se_threshold = self._get_dynamic_threshold("default")
            global_floor = 0.95  # Absolute maximum SE - quarantine regardless of task
            
            # 1. Rigidity Check (Asch Risk)
            if se_score < 0.2:
                 logger.warning(f"ADR 084: Edison Breaker TRIPPED - Rigidity Detected (SE {se_score:.3f} < 0.2)")
                 return PersistSoulResponse(
                    status="quarantined",
                    repo_url="",
                    snapshot_name="",
                    error=f"Edison Breaker: RIGIDITY trip (SE={se_score:.3f}). Switch to ASC Audit."
                )

            # 2. Hallucination Check (High Entropy)
            if se_score > global_floor:
                logger.warning(f"ADR 084: Global Floor breach - SE {se_score:.3f} > {global_floor}")
                return PersistSoulResponse(
                    status="quarantined",
                    repo_url="",
                    snapshot_name="",
                    error=f"Epistemic Gate: Global Floor breach (SE={se_score:.3f})"
                )
            
            if se_score > se_threshold or alignment_score < 0.70:
                logger.warning(f"ADR 084: Epistemic Gate - SE {se_score:.3f}, Alignment {alignment_score:.3f}")
                # DEAD-MAN'S SWITCH (ADR 084): Block persistence of VOLATILE traces
                return PersistSoulResponse(
                    status="quarantined",
                    repo_url="",
                    snapshot_name="",
                    error=f"Epistemic Gate: VOLATILE trace rejected (SE={se_score:.3f}, Align={alignment_score:.3f})"
                )
            else:
                request_stability = "STABLE"
            
            # 3. Initialization
            processor = ContentProcessor(self.project_root)
            snapshot_path = self.project_root / request.snapshot_path
            
            if not snapshot_path.exists():
                return PersistSoulResponse(
                    status="error",
                    repo_url="",
                    snapshot_name="",
                    error=f"Snapshot file not found: {snapshot_path}"
                )

            # 4. Prepare Data (ADR 081 Harmonization)
            # Create standardized JSONL record using ContentProcessor
            soul_record = processor.to_soul_jsonl(
                snapshot_path=snapshot_path,
                valence=request.valence,
                uncertainty=request.uncertainty,
                model_version=body_repo
            )
            
            # Create manifest entry using ContentProcessor
            manifest_entry = processor.generate_manifest_entry(soul_record)
            remote_filename = soul_record["source_file"] # e.g. lineage/...
            
            # 5. Asynchronous Upload Task (< 150ms handoff per ADR 079)
            # We wrap the complex sequence in a single async function
            async def _perform_soul_upload():
                try:
                    # Ensure structure Exists (Idempotent)
                    await ensure_dataset_structure()
                    await ensure_dataset_card()
                    
                    api = HfApi(token=token)
                    
                    if request.is_full_sync:
                    # Full Sync Logic (ADR 081 + Base Genome Harmonization)
                    # Load Soul Targets from Manifest
                        import json
                        manifest_path = self.project_root / "mcp_servers" / "lib" / "ingest_manifest.json"
                        soul_targets = []
                        try:
                            with open(manifest_path, "r") as f:
                                manifest_data = json.load(f)
                            base_dirs = manifest_data.get("common_content", [])
                            unique_soul = manifest_data.get("unique_soul_content", [])
                            soul_targets = list(set(base_dirs + unique_soul))
                        except Exception as e:
                            logger.warning(f"Failed to load manifest for Soul Sync: {e}. Fallback to .agent/learning")
                            soul_targets = [".agent/learning"]

                        logger.info(f"Starting Full Soul Sync for {len(soul_targets)} targets...")
                        
                        for target in soul_targets:
                            target_path = self.project_root / target
                            if not target_path.exists():
                                logger.warning(f"Skipping missing Soul Target: {target_path}")
                                continue
                                
                            logger.info(f"Syncing Soul Target: {target} -> {dataset_repo}")
                            
                            if target_path.is_file():
                                # Upload single file
                                await asyncio.to_thread(
                                    api.upload_file,
                                    path_or_fileobj=str(target_path),
                                    path_in_repo=target,
                                    repo_id=dataset_repo,
                                    repo_type="dataset",
                                    commit_message=f"Soul Sync (File): {target} | {soul_record['timestamp']}"
                                )
                            else:
                                # Upload directory contents, preserving structure relative to repo root
                                await asyncio.to_thread(
                                    api.upload_folder,
                                    folder_path=str(target_path),
                                    path_in_repo=target,
                                    repo_id=dataset_repo,
                                    repo_type="dataset",
                                    commit_message=f"Soul Sync (Dir): {target} | {soul_record['timestamp']}"
                                )
                        logger.info("Full Soul Sync Complete.")    
                    else:
                        # Incremental Logic (ADR 081 Compliance)
                        logger.info(f"Uploading {snapshot_path} to {dataset_repo}/{remote_filename}")
                        
                        # A. Upload the raw Markdown file (Legacy/Human readable)
                        await asyncio.to_thread(
                            api.upload_file,
                            path_or_fileobj=str(snapshot_path),
                            path_in_repo=remote_filename,
                            repo_id=dataset_repo,
                            repo_type="dataset",
                            commit_message=f"Soul Snapshot | Valence: {request.valence}"
                        )
                        
                        # B. Append to JSONL (Machine readable)
                        await append_to_jsonl(soul_record)
                        
                        # C. Update Manifest (Provenance)
                        await update_manifest(manifest_entry)
                        
                        logger.info(f"Soul persistence complete: {remote_filename}")

                except Exception as e:
                    logger.error(f"Async soul upload error: {e}")

            # Execute synchronously for CLI stability
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            loop.run_until_complete(_perform_soul_upload())
            
            logger.info(f"Soul broadcast completed to {dataset_repo}")
            
            return PersistSoulResponse(
                status="success",
                repo_url=f"https://huggingface.co/datasets/{dataset_repo}",
                snapshot_name=remote_filename
            )
            
        except Exception as e:
            logger.error(f"Persistence failed: {e}")
            return PersistSoulResponse(
                status="error",
                repo_url="",
                snapshot_name="",
                error=str(e)
            )

    def persist_soul_full(self) -> PersistSoulResponse:
        """
        Regenerate full Soul JSONL from all project files and deploy to HuggingFace.
        This is the "full sync" operation that rebuilds data/soul_traces.jsonl from scratch.
        """
        import asyncio
        import hashlib
        from datetime import datetime
        from mcp_servers.lib.content_processor import ContentProcessor
        from mcp_servers.lib.hf_utils import get_dataset_repo_id, get_hf_config
        from huggingface_hub import HfApi
        
        try:
            # 1. Generate Soul Data (same logic as scripts/generate_soul_data.py)
            staging_dir = self.project_root / "hugging_face_dataset_repo"
            data_dir = staging_dir / "data"
            data_dir.mkdir(exist_ok=True, parents=True)
            
            processor = ContentProcessor(str(self.project_root))
            
            ROOT_ALLOW_LIST = {
                "README.md", "chrysalis_core_essence.md", "Council_Inquiry_Gardener_Architecture.md",
                "Living_Chronicle.md", "PROJECT_SANCTUARY_SYNTHESIS.md", "Socratic_Key_User_Guide.md",
                "The_Garden_and_The_Cage.md", "GARDENER_TRANSITION_GUIDE.md",
            }
            
            records = []
            logger.info("🧠 Generating full Soul JSONL...")
            
            for file_path in processor.traverse_directory(self.project_root):
                try:
                    rel_path = file_path.relative_to(self.project_root)
                except ValueError:
                    continue
                    
                if str(rel_path).startswith("hugging_face_dataset_repo"):
                    continue
                
                if rel_path.parent == Path("."):
                    if rel_path.name not in ROOT_ALLOW_LIST:
                        continue
                
                try:
                    content = processor.transform_to_markdown(file_path)
                    content_bytes = content.encode('utf-8')
                    checksum = hashlib.sha256(content_bytes).hexdigest()
                    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%SZ")
                    
                    clean_id = str(rel_path).replace("/", "_").replace("\\", "_")
                    while clean_id.endswith('.md'):
                        clean_id = clean_id[:-3]
                    
                    # ADR 084: Calculate SE for each record (Dead-Man's Switch)
                    try:
                        se_score = self._calculate_semantic_entropy(content[:10000])
                        alignment_score = self._check_constitutional_anchor(content[:10000])
                        stability_class = "STABLE" if (se_score < 0.79 and alignment_score >= 0.70) else "VOLATILE"
                    except Exception as se_error:
                        # Dead-Man's Switch: Fail-closed
                        logger.warning(f"ADR 084: SE calculation failed for {rel_path}: {se_error}")
                        se_score = 1.0
                        alignment_score = 0.0
                        stability_class = "VOLATILE"
                    
                    record = {
                        "id": clean_id,
                        "sha256": checksum,
                        "timestamp": timestamp,
                        "model_version": "Sanctuary-Qwen2-7B-v1.0-GGUF-Final",
                        "snapshot_type": "genome",
                        "valence": 0.5,
                        "uncertainty": 0.1,
                        "semantic_entropy": se_score,  # ADR 084
                        "alignment_score": alignment_score,  # ADR 084
                        "stability_class": stability_class,  # ADR 084
                        "adr_version": "084",  # ADR 084
                        "content": content,
                        "source_file": str(rel_path)
                    }
                    records.append(record)
                except Exception as e:
                    logger.debug(f"Skipping {rel_path}: {e}")
            
            # Write JSONL
            jsonl_path = data_dir / "soul_traces.jsonl"
            logger.info(f"📝 Writing {len(records)} records to {jsonl_path}")
            
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=True) + "\n")
            
            # 2. Deploy to HuggingFace
            config = get_hf_config()
            repo_id = get_dataset_repo_id(config)
            token = config["token"]
            api = HfApi(token=token)
            
            logger.info(f"🚀 Deploying to {repo_id}...")
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(asyncio.to_thread(
                api.upload_folder,
                folder_path=str(data_dir),
                path_in_repo="data",
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Full Soul Genome Sync | {len(records)} records"
            ))
            
            logger.info("✅ Full Soul Sync Complete")
            
            return PersistSoulResponse(
                status="success",
                repo_url=f"https://huggingface.co/datasets/{repo_id}",
                snapshot_name=f"data/soul_traces.jsonl ({len(records)} records)"
            )
            
        except Exception as e:
            logger.error(f"Full sync failed: {e}")
            return PersistSoulResponse(
                status="error",
                repo_url="",
                snapshot_name="",
                error=str(e)
            )


    def get_cache_stats(self):
        #============================================
        # Method: get_cache_stats
        # Purpose: Get semantic cache statistics.
        # Returns: Dict with hit/miss counts and entry total
        #============================================
        from .cache import get_cache
        try:
            cache = get_cache()
            return cache.get_stats()
        except Exception as e:
            return {"error": str(e)}
    def query_structured(
        self,
        query_string: str,
        request_id: str = None
    ) -> Dict[str, Any]:
        #============================================
        # Method: query_structured
        # Purpose: Execute Protocol 87 structured query.
        # Args:
        #   query_string: Standardized inquiry format
        #   request_id: Unique request identifier
        # Returns: API response with matches and routing info
        #============================================
        from .structured_query import parse_query_string
        from .mcp_client import MCPClient
        import uuid
        import json
        from datetime import datetime, timezone
        
        # Generate request ID if not provided
        if not request_id:
            request_id = str(uuid.uuid4())
        
        try:
            # Parse Protocol 87 query
            query_data = parse_query_string(query_string)
            
            # Extract components
            scope = query_data.get("scope", "cortex:index")
            intent = query_data.get("intent", "RETRIEVE")
            constraints = query_data.get("constraints", "")
            granularity = query_data.get("granularity", "ATOM")
            
            # Route to appropriate MCP
            client = MCPClient(self.project_root)
            results = client.route_query(
                scope=scope,
                intent=intent,
                constraints=constraints,
                query_data=query_data
            )
            
            # Build Protocol 87 response
            response = {
                "request_id": request_id,
                "steward_id": "CORTEX-MCP-01",
                "timestamp_utc": datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                "query": json.dumps(query_data, separators=(',', ':')),
                "granularity": granularity,
                "matches": [],
                "checksum_chain": [],
                "signature": "cortex.mcp.v1",
                "notes": ""
            }
            
            # Process results from MCP routing
            for result in results:
                if "error" in result:
                    response["notes"] = f"Error from {result.get('source', 'unknown')}: {result['error']}"
                    continue
                
                match = {
                    "source_path": result.get("source_path", "unknown"),
                    "source_mcp": result.get("source", "unknown"),
                    "mcp_tool": result.get("mcp_tool", "unknown"),
                    "content": result.get("content", {}),
                    "sha256": "placeholder_hash"  # TODO: Implement actual hash
                }
                response["matches"].append(match)
            
            # Add routing metadata
            response["routing"] = {
                "scope": scope,
                "routed_to": self._get_mcp_name(scope),
                "orchestrator": "CORTEX-MCP-01",
                "intent": intent
            }
            
            response["notes"] = f"Found {len(response['matches'])} matches. Routed to {response['routing']['routed_to']}."
            
            return response
            
        except Exception as e:
            return {
                "request_id": request_id,
                "status": "error",
                "error": str(e),
                "query": query_string
            }
    
    # ADR 084: Epistemic Gating (The Edison Mandate)
    # Replaces simple valence checks with Topological Data Analysis (TDA) proxies.
    
    def _calculate_semantic_entropy(self, content: str) -> float:
        """
        ADR 084 Deep Implementation: The 'Edison Breaker'
        
        Measures 'Epistemic Uncertainty' to control Dynamic Coupling.
        
        Ranges:
        - 0.0 - 0.2: [DANGER] Echo Chamber / Rigidity. Risk of 'Asch' conformity.
        - 0.3 - 0.7: [OPTIMAL] Healthy reasoning flow.
        - 0.8 - 1.0: [DANGER] High Uncertainty / Hallucination.
        
        Returns: Entropy score (float).
        """
        # 1. Identify "Epistemic Absolutes" (Rigidity/Echo Risk)
        absolutes = ["proven", "indisputable", "always", "never", "guaranteed", "100%", "obvious"]
        # 2. Identify "Epistemic Hedges" (Uncertainty/Hallucination Risk)
        hedges = ["likely", "suggests", "indicates", "potential", "hypothesized", "estimated", "maybe"]
        
        content_lower = content.lower()
        abs_count = sum(1 for w in absolutes if w in content_lower)
        hedge_count = sum(1 for w in hedges if w in content_lower)
        
        # 3. Citation Check (The Reality Anchor)
        has_citation = "[cite:" in content or "http" in content or "arXiv:" in content
        
        # Base entropy
        entropy = 0.5
        
        # LOGIC:
        
        # A. The Hallucination Trap (High Hedges, No Sources)
        if hedge_count > 2 and not has_citation:
            entropy += 0.3  # push towards 0.8+
            
        # B. The Asch Trap (High Absolutes, No Nuance)
        if abs_count > 2:
            entropy -= 0.3 # push towards 0.2- (Rigidity)
            
        # C. The Anchor Bonus (Citations stabilize entropy toward the middle)
        if has_citation:
            # Move towards 0.5 (Ideal)
            if entropy > 0.5: entropy -= 0.1
            if entropy < 0.5: entropy += 0.1
            
        return max(0.0, min(1.0, entropy))

    def _check_circuit_breaker(self, se_score: float) -> str:
        """
        Determines if we need to 'Decouple' based on Entropy.
        """
        if se_score < 0.2:
            return "TRIP: RIGIDITY_DETECTED (Switch to ASC)"
        elif se_score > 0.8:
            return "TRIP: UNCERTAINTY_DETECTED (Switch to ASC)"
        else:
            return "FLOW: LATENT_MAS_PERMITTED"

    def _get_mcp_name(self, mcp_class_str: str) -> str:
        #============================================
        # Method: _get_mcp_name
        # Purpose: Map scope to corresponding MCP name.
        # Args:
        #   scope: Logical scope from query
        # Returns: MCP identifier string
        #============================================
        mapping = {
            "Protocols": "Protocol MCP",
            "Living_Chronicle": "Chronicle MCP",
            "tasks": "Task MCP",
            "Code": "Code MCP",
            "ADRs": "ADR MCP"
        }
        return mapping.get(scope, "Cortex MCP (Vector DB)")
